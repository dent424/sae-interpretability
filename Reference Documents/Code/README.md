# SAE Interpretability Notebook - Mexican Restaurant Reviews

This file documents `fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`, a Python script derived from a Colab notebook for analyzing Sparse Autoencoder (SAE) features extracted from Yelp restaurant reviews.

## Overview

The code implements an interpretability pipeline that:
1. Loads a pre-trained SAE model trained on GPT-2 residual stream activations
2. Extracts and analyzes features from 432,248 Mexican restaurant reviews
3. Provides tools for exploring what individual SAE features represent

## Model Architecture

- **Base Model**: GPT-2 (layer 8 residual stream, 768 dimensions)
- **SAE Type**: Top-K Sparse Autoencoder
- **Expansion Factor**: 32x (24,576 latent features)
- **Top-K**: 32 active features per token
- **Learning Rate**: 0.0003

## Code Structure

### Chapter 0: Setup and Data Loading
- Google Drive mounting and path configuration
- Installation of dependencies (transformer-lens, etc.)
- Loading of pre-computed SAE activations from H5 files
- Loading of review metadata from NPZ files

### Chapter 1: Core SAE Infrastructure

**TopKSAE Class** (`nn.Module`)
- `W_enc`: Encoder weights (768 x 24576)
- `W_dec`: Decoder weights (768 x 24576)
- `b_pre`: Pre-encoder bias (baseline activations)
- `k`: Number of top activations to keep (32)
- Forward pass: encode, apply top-k gating, decode

**Key Functions**:
- `get_gpt2_residuals()` - Extract residual stream activations from GPT-2 using transformer-lens hooks
- `process_new_text()` - Run arbitrary text through GPT-2 → SAE pipeline, returns activations and tokens
- `visualize_feature_activation()` - Generate HTML visualization of token-level feature activations
- `extract_feature_data_adapted()` - Extract example contexts for a given feature from the H5 dataset

### Chapter 2: Interpretability Playground
- Interactive exploration of individual features
- Example analyses for specific features of interest
- Context extraction and visualization workflows

### Chapter 3: Data Inspection
- Tools for examining H5 sparse storage format
- Metadata exploration utilities
- Feature activation statistics

## Data Format

### H5 Files (Sparse Activations)
- `z_idx`: Feature indices for non-zero activations
- `z_val`: Activation values
- `rev_idx`: Review boundary indices

### NPZ Files (Metadata)
- `review_ids`: Yelp review identifiers
- `texts`: Full review text
- `stars`: Star ratings (1-5)
- `useful`: Usefulness votes
- `user_ids`: Reviewer identifiers
- `business_ids`: Restaurant identifiers

## Notable Features Analyzed

| Feature | Description |
|---------|-------------|
| 16751 | Emphatic complaints/expressions |
| 20379 | Lexical pattern (gum/gumbo) |
| 14292 | Demonstrative phrases |
| 11328 | Formatted lists (bullets, numbered) |
| 23016 | Structural discourse markers |

## Dependencies

- torch
- transformer-lens
- numpy
- h5py
- pandas
- matplotlib
- IPython (for HTML display)

## Origin

Derived from research paper: "From Hidden Neural Representations to Actionable Insight" - using SAEs to extract interpretable features that predict review helpfulness.
