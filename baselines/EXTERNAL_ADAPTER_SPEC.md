# External Adapter Specification

External baselines should be treated as separate methods with thin adapters
around the shared Morpho-CellFlux data contract. Do not rewrite their model code
inside `src/morphoflux`.

## What each adapter must provide

1. **Data export**
   - Read a Morpho-CellFlux YAML config.
   - Load `data_index_path`, `image_path`, `embedding_path`, and `channels`.
   - Export train/test images and labels into the external method's expected
     dataset format.
   - Preserve `SAMPLE_KEY`, `CPD_NAME`, `BATCH`, `SPLIT`, and the sampled
     treated-to-control mapping.

2. **Condition mapping**
   - Diet: map `adlib`, `fasted`, `hfd` to condition labels or one-hot vectors.
   - CRISPR: use the same gene identity vocabulary as
     `embedding_gene_identity.csv`; for methods that require compact labels,
     provide a deterministic mapping from gene name to class index.
   - Record the mapping in the output directory.

3. **Training launcher**
   - Use method-native launchers/configs whenever possible.
   - Store all external checkpoints under
     `outputs/baselines/<method>/<benchmark>/external_checkpoints/` or a
     method-specific subdirectory.
   - For GPU baselines, use one job across both GPUs:
     `CUDA_VISIBLE_DEVICES=0,1`.

4. **Inference exporter**
   - Generate one output PNG per target row under:

     ```text
     outputs/baselines/<method>/<benchmark>/
       fid_samples/
         epoch-0/
           <condition_or_gene>/<target_sample_id>.png
         trt2ctrl_idx.json
     ```

   - Generated PNG channel order must match the config's `channels`.
   - `args.json` must include `channels`, `image_path`, `data_index_path`,
     `embedding_path`, method name, benchmark name, split, and checkpoint path.

5. **Evaluation**
   - Run the shared evaluator only:

     ```bash
     python scripts/aggregate_eval.py outputs/baselines/<method>/<benchmark> 5 0
     ```

   - Do not report method-native metrics in the main paper table unless the same
     metric is recomputed for every method.

## Proposed named baseline set

Primary 4-5 named baselines:

| Method | Paper line | Use in our table | Risk |
|---|---|---|---|
| PhenDiff | MICCAI 2024 cell phenotype diffusion translation | Main | adapter cost, many CRISPR classes |
| IMPA | Nature Communications 2025 image perturbation autoencoder | Main | GAN stability, data-format adapter |
| MorphoDiff | diffusion perturbation-encoding cell morphology generation | Main or supplement | may not use control image |
| StarGAN | classic multi-domain image translation | Supplement/main if results stable | older and less task-specific |
| CellFlux-style shared-panel | CellFlux-engine recipe adapted to the shared panel; not an original upstream CellFlux benchmark | Main | already available |

Optional if time permits:

| Method | Why optional |
|---|---|
| CycleGAN | only practical for diet pairwise translation, not many-class CRISPR |
| MorphGen | strong recent model, but task/data assumptions differ and adapter may be heavy |
| MorphDiff | transcriptome-guided; needs clean transcriptome condition, not always available for diet |
| no-control FM | useful mechanistic ablation, but not a named external method |
