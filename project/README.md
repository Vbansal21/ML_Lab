# Stock Price Prediction System

A modular PyTorch-based system for time series prediction focused on stock market data using transformer models and weight-sharing techniques.

## Features

- **Configurability**: YAML-based configuration system
- **Modular Architecture**: Separation of concerns between data, model, and training components
- **Advanced Model**: Transformer autoencoder with weight sharing and custom positional embeddings
- **Multi-Stock Support**: Predicts multiple stocks with a unified model
- **Technical Indicators**: Calculates and uses various technical indicators as features
- **Comprehensive Metrics**: Detailed evaluation with per-ticker performance analysis
- **Visualization**: Creates plots for training progress and prediction results

## Project Structure

- `config.yaml`: Configuration file with all parameters and settings
- `main.py`: Main execution script with command-line interface
- `utils.py`: General utility functions for logging, configuration, etc.
- `data_utils.py`: Data loading, processing, and sequence preparation 
- `model_utils.py`: Model architecture definitions

## Setup

### Dependencies

```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm yfinance pyyaml
```

### Configuration

All parameters are defined in `config.yaml`. You can customize:
- Tickers to analyze
- Date ranges
- Model architecture
- Training parameters
- Technical indicators to calculate
- Output and logging settings

## Usage

### Training
```bash
python main.py --mode train
```

### Making Predictions
```bash
python main.py --mode predict --checkpoint path/to/model_checkpoint.pth
```

### Evaluating a Model
```bash
python main.py --mode evaluate --checkpoint path/to/model_checkpoint.pth
```

### Command-line Arguments
- `--config`: Path to configuration file (default: 'config.yaml')
- `--mode`: Mode to run: 'train', 'predict', or 'evaluate' (default: 'train')
- `--tickers`: Override tickers in config
- `--start_date`: Override start date in config
- `--end_date`: Override end date in config
- `--output_dir`: Override output directory in config
- `--checkpoint`: Model checkpoint to load for evaluation/prediction

## Weight Sharing Structure

The model uses a weight sharing scheme where transformer blocks can share weights using the configuration pattern:
```yaml
block_weight_indices: [0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 6]
```

This creates a 15-layer network using only 7 unique transformer blocks, where:
- The first layer (layer 0) uses block 0
- Layers 1-5 share the same parameters (block 1)
- Layer 6 uses block 2
- Layer 7 uses block 3 (before bottleneck)
- Layer 8 uses block 4 (after bottleneck)
- Layers 9-13 share the same parameters (block 5)
- The final layer (layer 14) uses block 6

This pattern provides independent weights for layers around the bottleneck while using parameter sharing elsewhere for efficiency.

## Model Architecture

The model is a transformer-based autoencoder that:
1. Embeds stock tickers into a vector space
2. Merges ticker embeddings with feature data
3. Processes the sequence through transformer blocks
4. Creates a latent representation at the bottleneck
5. Reconstructs the input to predict the next timestep

The model uses causal masking to ensure predictions only use information from previous timesteps, and can optionally apply XPos positional embeddings to improve sequence modeling. 