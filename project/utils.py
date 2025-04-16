import yaml
import logging
import os
import torch
import numpy as np
import random
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Loads configuration settings from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise

def setup_logging(log_config, log_dir):
    """Configures logging based on the provided configuration."""
    log_level = getattr(logging, log_config['log_level'].upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()] # Log to console by default

    if log_config['log_to_file']:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_config['log_filename'])
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.info(f"Logging setup complete. Level: {log_config['log_level']}. Log file: {log_file if log_config['log_to_file'] else 'Console only'}")

def set_seed(seed):
    """Sets the random seed for reproducibility across libraries."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Potentially make CuDNN deterministic (can impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logging.info(f"Random seed set to {seed}")
    else:
        logging.info("No random seed specified.")

def get_device(device_config):
    """Determines the computing device (CPU or CUDA) based on config and availability."""
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logging.warning("CUDA device requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    return device

def create_output_dirs(config):
    """Creates all necessary output directories defined in the config."""
    base_dir = Path(config['paths']['output_dir'])
    plot_dir = base_dir / config['paths']['plot_subdir']
    log_dir = base_dir / config['paths']['log_subdir']
    tensorboard_dir = base_dir / config['paths']['tensorboard_subdir']

    base_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    logging.info(f"Created output directories in {base_dir}")
    return {
        "base": base_dir,
        "plots": plot_dir,
        "logs": log_dir,
        "tensorboard": tensorboard_dir
    }

# Add more utility functions as needed, e.g., for saving/loading checkpoints 