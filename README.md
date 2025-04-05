# Online Learning Differentiable Particle Filters (OL-DPF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:

**"Learning Differentiable Particle Filter on the Fly"** (57th Asilomar Conference on Signals, Systems, and Computers, 2023)

## Repository Structure

The repository contains two main folders, each corresponding to a different experimental setup:

### OL-DPF-gaussian (Multivariate Linear Gaussian Model)
### OL-DPF-tracking (Non-linear Object Tracking Model)

Each folder is self-contained with the following structure:

```
.
├── arguments.py        # Parameter configuration
├── inference.py        # Inference logic and particle propagation
├── losses.py           # Loss functions (RMSE, ELBO)
├── train.py            # Training loop management
├── state.py            # Particle state management
├── aemath.py           # Mathematical utilities
├── statistics.py       # Statistical estimation methods
├── test_losses.py      # Core script for launching experiments
├── lgssm.py            # State-space model implementations
├── resamplers/
│   └── resamplers.py   # Particle resampling implementations
└── setup.py            # Installation script
```

## Installation

### Environment Requirements
- Ubuntu 22.04 LTS (or other Linux distributions)
- Python 3.7
- CUDA-compatible GPU recommended (experiments were conducted on NVIDIA GeForce RTX 3090)

### Setup
```bash
# Create and activate a Python 3.7 environment
conda create -n ol-dpf python=3.7
conda activate ol-dpf

# Clone the repository
git clone https://github.com/JiaxiLi1/Online-Learning-Differentiable-Partiacle-Filters.git
cd Online-Learning-Differentiable-Partiacle-Filters

# Navigate to the experiment directory of interest
cd OL-DPF-gaussian  # or cd OL-DPF-tracking

# Install required packages
pip install .
```

## Experiments

The repository supports experiments on two types of datasets:

### 1. Multivariate Linear Gaussian Model

```bash
# Navigate to the Gaussian model directory
cd OL-DPF-gaussian

# Run the experiment
python test_losses.py --trainType [online/offline] --num_dim [2/5/10] --resampler_type [multinomial/stratified/systematic] --NF-dyn [True/False] --NF-cond [True/False] --device [cuda/cpu]
```

### 2. Non-linear Object Tracking Model

```bash
# Navigate to the tracking model directory
cd OL-DPF-tracking

# Run the experiment
python test_losses.py --trainType [online/offline] --resampler_type [multinomial/stratified/systematic] --NF-dyn [True/False] --NF-cond [True/False] --device [cuda/cpu] --measurement [type]
```

### Configuration Parameters

- `trainType`: Selects between online or offline
- `num_dim`: Dimensionality of the latent state and observation vectors (for Gaussian model, we tested d=2, 5, 10)
- `resampler_type`: Particle resampling method
- `NF-dyn`: Enable/disable normalizing flows for the dynamic model
- `NF-cond`: Enable/disable normalizing flows for the proposal distribution
- `device`: Computational device (cuda for GPU, cpu for CPU)
- `measurement`: Measurement model type

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@Inproceedings{li2023,
  title={Learning Differentiable Particle Filter on the Fly},
  author={Li, Jiaxi, Chen, Xiongjie and Li, Yunpeng},
  booktitle={57th Asilomar Conference on Signals, Systems, and Computers},
  year={2023}
}
```

## Acknowledgments

This codebase builds upon the implementation from:
```bibtex
@Inproceedings{le2018auto,
	title={Auto-Encoding sequential {Monte Carlo}},
	author={Le, T. A. and Igl, M. and Rainforth, T. and Jin, T. and Wood, F.},
	booktitle={Proc. Int. Conf. Learn. Rep. (ICLR)},
	address={Vancouver, Canada},
	month={Apr.},
	year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
