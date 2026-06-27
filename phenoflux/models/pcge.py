"""
PhenoFlux PCGE: Hierarchical Program-Conditioned Gene Embedding for CRISPR.

GeneProgramEncoder: replaces the flat 204-dim one-hot gene identity with
a hierarchical embedding that captures program-level structure, enabling
information sharing across functionally related genes.

Each gene gets:
  gene_embedding + weighted_sum(program_prototypes) -> condition vector

Program prototypes (K=7) are learnable.  Genes in the same program (e.g.
all UPR genes) share a program-level representation, so the model can
learn "UPR program -> Calreticulin up" from observing multiple UPR genes.

build_gene_to_program_mapping: reads index_paper_programs.csv and builds
a frozen gene_index -> program_id mapping used by GeneProgramEncoder.

Reference:
  PhenoFlux: Marker-Aware Flow Matching for Molecular Phenotype Transport
"""

import logging
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeneProgramEncoder -- nn.Embedding subclass with program-level structure
# ---------------------------------------------------------------------------


class GeneProgramEncoder(nn.Embedding):
    """Hierarchical gene embedding with program-conditioned cross-attention.

    Inherits nn.Embedding for drop-in compatibility with the existing
    datamodule.embedding_matrix.  Replaces flat one-hot lookup with:

    1. F.embedding(gene_index, self.weight)           -> gene_emb (B, E)
    2. Cross-attention to K learnable program prototypes -> prog_ctx (B, E)
    3. Gated fusion: gate * gene_emb + (1-gate) * prog_ctx -> combined
    4. Linear projection to output_dim (matches original one-hot dim)

    Genes without a program (gene_to_prog == num_programs) use pure gene
    embedding with no program contribution.

    Parameters
    ----------
    num_genes : int
        Number of unique gene identities (len(mol_names)).
    embed_dim : int
        Internal embedding dimension (default 256).
    output_dim : int
        Output dimension matching original one-hot condition_dim.
    num_programs : int
        Number of biological programs (typically K=7).
    gene_to_program : LongTensor or None
        Frozen (num_genes,) buffer mapping gene_index -> program_id (0..K-1)
        or num_programs (sentinel for "no program").
    padding_idx : int or None
        Passed to nn.Embedding for padding support.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 256,
        output_dim: int = 204,
        num_programs: int = 7,
        gene_to_program: torch.LongTensor | None = None,
        padding_idx: int | None = None,
    ):
        super().__init__(num_genes, embed_dim, padding_idx=padding_idx)
        self.num_programs = num_programs
        self.output_dim = output_dim

        # K learnable program prototypes
        self.program_proto = nn.Parameter(
            torch.randn(num_programs, embed_dim) * 0.02
        )

        # Gated fusion: balances gene-specific vs program-shared information
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Output projection: embed_dim -> output_dim (matches original condition_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # Cross-attention temperature (sharp start, softens over training)
        self.temperature = nn.Parameter(torch.tensor(0.1))

        # Frozen gene-to-program mapping (buffer: saved/loaded, not learned)
        self.register_buffer(
            'gene_to_prog',
            gene_to_program
            if gene_to_program is not None
            else torch.full((num_genes,), num_programs, dtype=torch.long),
        )

    def forward(self, gene_indices: torch.LongTensor) -> torch.Tensor:
        """Lookup gene embedding with program-conditioned augmentation.

        Args:
            gene_indices: (B,) LongTensor of gene identity indices.

        Returns:
            (B, output_dim) conditioning vector matching original one-hot dim.
        """
        # 1. Base embedding lookup (identical to nn.Embedding.forward)
        gene_emb = F.embedding(
            gene_indices,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )  # (B, embed_dim)

        # 2. Program assignment from frozen buffer
        prog_idx = self.gene_to_prog[gene_indices]  # (B,)

        # 3. Cross-attention: each gene queries all K program prototypes
        attn_scores = (
            gene_emb @ self.program_proto.T
        ) / self.temperature.abs()  # (B, K)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, K)
        prog_ctx = attn_weights @ self.program_proto  # (B, embed_dim)

        # 4. Gated fusion: balance gene-specific vs program-shared
        gate_input = torch.cat([gene_emb, prog_ctx], dim=-1)  # (B, 2*E)
        gate_val = self.gate(gate_input)  # (B, 1)
        combined = gate_val * gene_emb + (1.0 - gate_val) * prog_ctx

        # 5. Genes without a program: use pure gene embedding
        no_prog_mask = prog_idx == self.num_programs  # (B,)
        combined[no_prog_mask] = gene_emb[no_prog_mask]

        # 6. Project to output dimension
        return self.output_proj(combined)  # (B, output_dim)


# ---------------------------------------------------------------------------
# build_gene_to_program_mapping
# ---------------------------------------------------------------------------


def build_gene_to_program_mapping(
    index_csv_path: str,
    mol_names: List[str],
) -> Tuple[torch.LongTensor, int, List[str]]:
    """Build gene_index -> program_id mapping from Perturb-Multi index CSV.

    Reads index_paper_programs.csv, extracts CPD_NAME -> PROGRAM_ID pairs
    for treated genes with program annotations, and maps them to the
    mol2id embedding index order.

    Each gene maps to exactly one program.  Genes not in the index CSV
    (housekeeping, controls) get the sentinel value num_programs.

    Args:
        index_csv_path: path to index_paper_programs.csv (self.data_index_path).
        mol_names: list of gene names in mol2id order (from np.unique).

    Returns:
        gene_to_prog: (len(mol_names),) LongTensor.
            gene_to_prog[i] = program_id (0..K-1) or num_programs (sentinel).
        num_programs: int, number of unique programs (typically 7).
        program_names: sorted list of program name strings.
    """
    df = pd.read_csv(index_csv_path, index_col=0)

    # Only treated samples carry PROGRAM_ID information
    treated = df[df['ANNOT'] == 'treated']
    treated_with_prog = treated[treated['PROGRAM_ID'].notna()]

    mol_to_idx = {mol: i for i, mol in enumerate(mol_names)}

    program_names = sorted(treated_with_prog['PROGRAM'].unique())
    num_programs = len(program_names)

    gene_to_prog = torch.full(
        (len(mol_names),), num_programs, dtype=torch.long
    )

    mapped = 0
    for _, row in treated_with_prog.iterrows():
        gene_name = row['CPD_NAME']
        prog_id = int(row['PROGRAM_ID'])
        if gene_name in mol_to_idx:
            gene_to_prog[mol_to_idx[gene_name]] = prog_id
            mapped += 1

    logger.info(
        "PCGE: mapped %d genes to %d programs (%s)",
        mapped, num_programs, ', '.join(program_names),
    )
    return gene_to_prog, num_programs, program_names
