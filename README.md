
# Generative ML Models for RNA Editing

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Codecov](https://img.shields.io/codecov/c/github/username/repo)
![PyPI Version](https://img.shields.io/pypi/v/your_package_name)



This repository contains implementations of **generative machine learning models** designed for **RNA editing applications**, with a focus on ADAR-mediated guide RNA (gRNA) design. The primary objective is to explore and develop conditional generative models that can design gRNA sequences with desired structural and energetic properties.

---

## Features
- **Sequence Representation**: Efficient encoding and decoding of RNA and gRNA sequences.
- **Conditional Generative Models**:
  - Bit Diffusion Model (based on DNA-Diffusion & DeepREAD concepts)
  - Variational Autoencoders (VAE)
  - Transformer-based architectures
- **Conditioning Parameters**:
  - Target RNA sequence context
  - Predicted ΔG minimum free energy (MFE) for structure optimization
- **Training & Evaluation Pipeline**:
  - Stratified dataset creation
  - Conditional training with ΔG normalization
  - ΔG prediction head for supervised alignment
  - Model evaluation on ΔG and sequence fidelity

---

## Project Structure
```
Generative_ML_Models_for_RNA_editing/
│
├── data/                        # RNA/gRNA datasets (e.g., vienna_rna_full_features.csv)
|
├── models/                      # Model architectures (Diffusion, VAE, Transformer)
│   ├── bit_diffusion.py         # Core Bit Diffusion model
│
├── training/                    
│   ├── train.py               # Conditional training script
│   └── sample.py              # Sequence generation & inference
|   └── evaluation.py          # To evalute and generat all the box plots and other graphs
│
├── notebooks/                 # Cretaing the .pt files of the data set used for training

└── README.md
```



---

## References
1. [DeepREAD Paper](https://www.biorxiv.org/lookup/doi/10.1101/2024.09.27.613923)  
2. [DNA-Diffusion GitHub](https://github.com/pinellolab/DNA-Diffusion)  


