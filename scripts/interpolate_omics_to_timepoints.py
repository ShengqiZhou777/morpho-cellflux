#!/usr/bin/env python3
"""
Interpolate omics PCA data from 9 hourly timepoints to 105 5-minute bins.

Input:
  - /home/shockley/myproject/FusionODE/data/omics_bulk.csv (54 rows: 9 time × 2 cond × 3 rep)
  - data/processed/microalgae_v1/views/timepoint_512/embedding.csv (105 rows: timegroup bins)

Output:
  - data/processed/microalgae_v1/views/timepoint_512/embedding_61d.csv (105 rows, 61 dims)
    Columns: timegroup_key, cond_light, cond_dark, time_norm, time_bin_h,
             rna_pca_0...28, prot_pca_0...28
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def main():
    # Load existing data
    omics_pca = pd.read_csv('/home/shockley/myproject/FusionODE/data/omics_bulk.csv')
    embedding_4d = pd.read_csv('data/processed/microalgae_v1/views/timepoint_512/embedding.csv')

    print(f"=== Input Data ===")
    print(f"Omics PCA: {omics_pca.shape} (9 time × 2 cond × 3 rep)")
    print(f"Embedding 4D: {embedding_4d.shape} (105 timegroup bins)")

    # Extract PCA columns
    rna_pca_cols = [c for c in omics_pca.columns if c.startswith('rna_pca_')]
    prot_pca_cols = [c for c in omics_pca.columns if c.startswith('prot_pca_')]
    print(f"RNA PCA dims: {len(rna_pca_cols)}, Protein PCA dims: {len(prot_pca_cols)}")

    # Separate by condition
    dark_data = omics_pca[omics_pca['condition'] == 'Dark'].copy()
    light_data = omics_pca[omics_pca['condition'] == 'Light'].copy()

    print(f"\nDark samples: {len(dark_data)} (9 time × 3 rep)")
    print(f"Light samples: {len(light_data)} (9 time × 3 rep)")
    print(f"Timepoints: {sorted(dark_data['time'].unique())}")

    # Average across replicates for each timepoint
    def avg_by_time(df, cols):
        """Average PCA values across replicates for each timepoint."""
        return df.groupby('time')[cols].mean().reset_index()

    dark_avg = avg_by_time(dark_data, rna_pca_cols + prot_pca_cols)
    light_avg = avg_by_time(light_data, rna_pca_cols + prot_pca_cols)

    print(f"\nAfter averaging replicates:")
    print(f"Dark: {dark_avg.shape}, Light: {light_avg.shape}")

    # Build interpolators for each condition and each PCA dimension
    def build_interpolators(df, cols):
        """Build cubic spline interpolators for each PCA column."""
        times = df['time'].values
        interpolators = {}
        for col in cols:
            values = df[col].values
            # Cubic spline with natural boundary conditions
            interpolators[col] = interp1d(
                times, values, kind='cubic',
                fill_value='extrapolate', bounds_error=False
            )
        return interpolators

    dark_interp = build_interpolators(dark_avg, rna_pca_cols + prot_pca_cols)
    light_interp = build_interpolators(light_avg, rna_pca_cols + prot_pca_cols)

    # Interpolate for each timegroup bin
    embedding_61d = embedding_4d.copy()

    for col in rna_pca_cols + prot_pca_cols:
        embedding_61d[col] = np.nan

    for idx, row in embedding_61d.iterrows():
        time_h = row['time_bin_h']
        is_light = row['cond_light'] == 1.0

        # Choose interpolator based on condition
        interp_dict = light_interp if is_light else dark_interp

        # Interpolate each PCA dimension
        for col in rna_pca_cols + prot_pca_cols:
            embedding_61d.at[idx, col] = float(interp_dict[col](time_h))

    # Reorder columns: metadata first, then RNA PCA, then Protein PCA
    # ── Normalize PCA columns to z-score ──────────────────────────
    # Raw PCA values span [-486, 593] while base dims (light/dark/time_norm)
    # are in [0, 1]. This 500x scale mismatch would cause the condition Linear
    # layer to be dominated by PCA noise, pushing the model back to identity
    # mapping. Z-score each PCA column to mean=0, std=1 (~same scale as base dims).
    pca_cols = rna_pca_cols + prot_pca_cols
    pca_mean = embedding_61d[pca_cols].mean()
    pca_std = embedding_61d[pca_cols].std().replace(0.0, 1.0)  # avoid div-by-zero

    embedding_61d[pca_cols] = (embedding_61d[pca_cols] - pca_mean) / pca_std

    # Persist normalization stats for inference-time reuse
    stats_path = 'data/processed/microalgae_v1/views/timepoint_512/embedding_61d_stats.csv'
    stats_df = pd.DataFrame({'mean': pca_mean, 'std': pca_std})
    stats_df.to_csv(stats_path)
    print(f"\n=== Normalization ===")
    print(f"PCA z-scored to mean=0, std=1")
    print(f"After norm: min={embedding_61d[pca_cols].min().min():.2f}, max={embedding_61d[pca_cols].max().max():.2f}")
    print(f"Stats saved: {stats_path}")

    # Reorder columns: metadata first, then RNA PCA, then Protein PCA
    col_order = ['timegroup_key', 'cond_light', 'cond_dark', 'time_norm', 'time_bin_h'] + \
                rna_pca_cols + prot_pca_cols
    embedding_61d = embedding_61d[col_order]

    # Save
    out_path = 'data/processed/microalgae_v1/views/timepoint_512/embedding_61d.csv'
    embedding_61d.to_csv(out_path, index=False)

    print(f"\n=== Output ===")
    print(f"Saved: {out_path}")
    print(f"Shape: {embedding_61d.shape}")
    print(f"Columns: {list(embedding_61d.columns[:8])} ... {list(embedding_61d.columns[-3:])}")
    print(f"\nFirst 3 rows (first 10 cols):")
    print(embedding_61d.iloc[:3, :10])

    # Sanity check: verify interpolation at known timepoints
    # (de-normalize interpolated value back to raw scale for comparison)
    print(f"\n=== Sanity Check (de-normalized) ===")
    for time in [0.0, 1.0, 3.0, 6.0, 24.0]:
        # Find closest timegroup bin
        closest_idx = (embedding_61d['time_bin_h'] - time).abs().idxmin()
        closest_row = embedding_61d.loc[closest_idx]

        # Compare with original PCA value
        cond = 'Dark' if closest_row['cond_dark'] == 1.0 else 'Light'
        original = dark_avg if cond == 'Dark' else light_avg
        orig_row = original[original['time'] == time]

        if len(orig_row) > 0:
            orig_val = orig_row.iloc[0]['rna_pca_0']
            # De-normalize: interp_norm * std + mean
            interp_norm = closest_row['rna_pca_0']
            interp_val = interp_norm * pca_std['rna_pca_0'] + pca_mean['rna_pca_0']
            print(f"{cond} {time}h: Original rna_pca_0={orig_val:.2f}, Interpolated={interp_val:.2f}, Δ={abs(orig_val-interp_val):.3f}")

if __name__ == '__main__':
    main()
