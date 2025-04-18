# Normalizing Flows Implementation

This project is an interactive implementation of different Normalizing Flows models with a Streamlit user interface. It allows you to visualize and experiment with probabilistic transformations like NICE, RealNVP, and Glow on various data distributions.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Implemented Models](#implemented-models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Technical Aspects](#technical-aspects)
- [Contributions](#contributions)
- [References](#references)

## 🔍 Introduction

Normalizing Flows are generative models that learn to transform a simple distribution (like a Gaussian) into a complex distribution through a sequence of invertible transformations. This project offers an interactive interface to explore these models, understand how they work, and visualize their transformations.

## 🧠 Implemented Models

### NICE (Non-linear Independent Components Estimation)
- Uses additive couplings to transform distributions
- Simplified architecture with explicit invertible transformations
- Implementation based on the paper ["NICE: Non-linear Independent Components Estimation"](https://arxiv.org/abs/1410.8516)

### RealNVP (Real-valued Non-Volume Preserving)
- Extension of NICE with affine transformations (multiplication and addition)
- Allows more expressive mappings through scale changes
- Based on the paper ["Density Estimation using Real NVP"](https://arxiv.org/abs/1605.08803)

### Glow
- Advanced architecture combining invertible 1x1 convolutions and affine couplings
- Includes batch normalization and deeper networks
- Implementation according to the paper ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/EpsilonOF/Normalizing-flows-implementation.git
cd Normalizing-flows-implementation

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

To launch the Streamlit application:

```bash
streamlit run app.py
```

The web interface should automatically open in your browser, allowing you to:
- Select a model (NICE, RealNVP, Glow)
- Choose a data distribution (Circles, Spiral, Multiple Gaussians, etc.)
- Adjust training hyperparameters
- Visualize transformations in real-time

## 📁 Project Structure

```
Normalizing-flows-implementation/
├── app.py                  # Entry point for the Streamlit application
├── requirements.txt        # Project dependencies
├── presentation.py         # Subject presentation
├── nice/                   # NICE model implementation
│   ├── nice.py             # Model layout
│   ├── train_nice.py       # Model training
│   └── images/             # If images need to be saved
├── realnvp/                # RealNVP model implementation
│   ├── real_nvp.py         # Model layout
│   ├── train_realnvp.py    # Model training
│   └── images/             # If images need to be saved
├── glow/                   # Glow model implementation
│   ├── glow.py             # Model layout
│   ├── train_glow.py       # Model training
│   └── images/             # If images need to be saved
```

## ✨ Features

- **Real-time visualization**: Observe how the distribution transforms during training
- **Hyperparameter customization**: Adjust learning rate, batch size, number of epochs, etc.
- **Comparative analysis**: Compare the performance of different models on the same data

## 🔧 Technical Aspects

### Model Architecture

The models are implemented as PyTorch modules inheriting from a common base class, providing a consistent interface for:
- Direct transformation (forward) from a simple distribution to a complex distribution
- Inverse transformation (backward) to generate new samples
- Calculation of the log-determinant Jacobian for density estimation

### Training

Training uses maximum likelihood estimation as the objective:
- Minimize the KL divergence between the target distribution and the transformed distribution
- Optimization through stochastic gradient descent with Adam
- Tracking of training metrics such as negative log-likelihood

## Documentation with Sphinx

This project uses Sphinx to generate comprehensive and navigable documentation. Here's how to set up, create, and compile the documentation.

### Installing Sphinx

To install Sphinx and the necessary extensions, run:

```bash
pip install sphinx sphinx-rtd-theme autodoc numpydoc
```

### Documentation Structure

The documentation is organized in the `docs/` folder with the following structure:

```
docs/
├── source/
│   ├── _static/
│   ├── _templates/
│   ├── api/
│   ├── tutorials/
│   ├── conf.py
│   ├── index.rst
│   └── ...
├── Makefile
└── make.bat
```

### Generating Documentation

To automatically generate the documentation:

```bash
cd docs
make html
```

You will then find the HTML documentation in `docs/build/html/`.

## 📚 References

- [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803)
- [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)
- [Normalizing Flow implementation in PyTorch](https://github.com/VincentStimper/normalizing-flows)
