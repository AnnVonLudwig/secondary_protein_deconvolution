## Project Overview

This directory hosts the full workflow for photothermal spectrum deconvolution and protein component analysis. It combines traditional physics/mathematics techniques with deep learning: first performing a quick NNLS-based breakdown of experimental spectra, then training an early-fusion CNN on synthetic data to estimate ratios such as `Beta/Alpha`.

## Notebook Guide

- **`notebooks/nnls_analyze_spectrum.ipynb`**
  - Applies Non-negative Least Squares (NNLS) as a classical physics/mathematical method to the experimental spectrum.
  - Provides fast approximations of component contributions, guiding later simulation and modeling.
- **`notebooks/generate.ipynb`** / **`notebooks/generate3.ipynb`**
  - Create idealized spectral datasets according to experimental settings, supporting batch synthesis.
  - Tuned via configuration files such as `notebooks/generate_dataset3.yml` to control wavelength ranges, noise, component ratios, and other variables.
- **`notebooks/EarlyFusion+betterCNN4_1360_1800.ipynb`**
  - Trains and evaluates an early-fusion CNN on the generated datasets.
  - Requires the synthetic training set above and outputs metrics like the `Beta/Alpha` ratio.
- **`notebooks/predict_real_spectrum_with_early_fusion.ipynb`**
  - Loads the trained early-fusion CNN model and performs inference on real experimental spectra.

## Recommended Workflow

1. **Initial Analysis**: Run `nnls_analyze_spectrum.ipynb` to perform NNLS decomposition of the experimental spectrum and establish a coarse component estimate.
2. **Synthetic Dataset Generation**: Adjust `generate_dataset3.yml` and related parameters, then execute `generate.ipynb` or `generate3.ipynb` to produce training data.
3. **Model Training**: Use `EarlyFusion+betterCNN4_1360_1800.ipynb` to train the early-fusion CNN with the generated dataset.
4. **Model Inference**: Apply `predict_real_spectrum_with_early_fusion.ipynb` to evaluate real spectra and retrieve the `Beta/Alpha` ratio.

## Configuration and Dependencies

- Project dependencies are listed in `requirements.txt`; set up an isolated environment before installation.
- Supporting data, models, and physics routines live in `data/`, `models/`, `physics/`, and `data_process/` for deeper inspection.

## Usage Tips

- Before generating datasets, ensure the configuration files match the wavelength range and experimental conditions.
- Treat the NNLS results as a baseline reference when designing synthetic data and training strategies.
- Training can be time-consuming; save model checkpoints for future inference sessions.

## EarlyFusion+betterCNN4_1360_1800 Model Details

`notebooks/EarlyFusion+betterCNN4_1360_1800.ipynb` implements an early-fusion architecture that mixes raw spectra with engineered descriptors:

- **Dual-branch encoder**: The spectrum branch passes a normalized 1D trace through four wide convolutional layers (`1→128→256→512→512`, kernel sizes `11/7/5/3`) with ReLU activations, followed by adaptive average pooling to a fixed 32-step sequence. Parallel to that, the feature branch feeds concatenated integral/decomposition/NNLS descriptors into a 3-layer MLP (`64→32→16`) with dropout for regularization.
- **Self-attention refinement**: The pooled spectral embeddings enter a 4-head self-attention block (`MultiHeadSelfAttention1D`) with residual + layer norm, allowing the network to capture long-range correlations across wavenumbers before flattening to a 64-d latent embedding.
- **Early fusion head**: The 64-d CNN embedding and the 16-d tabular embedding are concatenated and processed by a final MLP (`(64+16)→32→1`) to regress the target `log1p(β/α)` ratio.
- **Training loop**: Datasets are wrapped in `EarlyFusionDataset`, batching spectra and features together. Training uses Adam (lr `1e-3`) with MSE loss over 400 epochs, early stopping (patience 40, min delta `1e-5`), and validation monitoring. Targets are exponentiated via `expm1` when reporting metrics or exporting predictions.
- **Outputs and persistence**: The notebook logs MAE/R² on validation and test splits, computes binned errors, and saves the best-performing weights plus normalization stats and dataset metadata for downstream inference.

