# Principal Neighborhood Aggregation (PNA) Model for Graph-Level Attribute Prediction

This project investigates whether Principal Neighbourhood Aggregation (PNA) improves message-passing graph neural networks on continuous node features and varied graph structures. It reproduces and compares PNA variants with conventional MPNNs on four graph-level prediction benchmarks: ZINC, CIFAR10 Superpixels, MNIST Superpixels, and OGBG-MolHIV.

The implementation and experiment outputs are in [`code.ipynb`](code.ipynb). The accompanying project report is not included in this repository.

## Methods

The PNA layers combine four neighbourhood aggregators (mean, standard deviation, maximum, and minimum) with degree-based amplification and attenuation scalers. The experiments compare PNA with scalers, PNA without scalers, and MPNNs using sum or maximum aggregation. For ZINC and the image benchmarks, variants are evaluated with and without edge features where applicable. MolHIV experiments use node features only.

## Benchmarks and reported results

The table below gives the results from this project as reported in the accompanying project report. ZINC uses mean absolute error (lower is better); CIFAR10 and MNIST use accuracy (higher is better); MolHIV uses ROC-AUC (higher is better). Values are single-run results.

| Dataset | Model | Edge features | Metric | Result |
| --- | --- | --- | --- | ---: |
| ZINC | MPNN (sum) | No | MAE | 0.3302 |
| ZINC | MPNN (max) | No | MAE | 0.4364 |
| ZINC | PNA (no scalers) | No | MAE | 0.3945 |
| ZINC | PNA | No | MAE | 0.2594 |
| ZINC | MPNN (sum) | Yes | MAE | 0.2569 |
| ZINC | MPNN (max) | Yes | MAE | 0.2755 |
| ZINC | PNA (no scalers) | Yes | MAE | 0.2006 |
| ZINC | PNA | Yes | MAE | 0.1799 |
| CIFAR10 Superpixels | MPNN (sum) | No | Accuracy | 61.94% |
| CIFAR10 Superpixels | MPNN (max) | No | Accuracy | 68.46% |
| CIFAR10 Superpixels | PNA (no scalers) | No | Accuracy | 70.02% |
| CIFAR10 Superpixels | PNA | No | Accuracy | 69.14% |
| CIFAR10 Superpixels | MPNN (sum) | Yes | Accuracy | 63.63% |
| CIFAR10 Superpixels | MPNN (max) | Yes | Accuracy | 69.87% |
| CIFAR10 Superpixels | PNA (no scalers) | Yes | Accuracy | 69.42% |
| CIFAR10 Superpixels | PNA | Yes | Accuracy | 68.35% |
| MNIST Superpixels | MPNN (sum) | No | Accuracy | 96.15% |
| MNIST Superpixels | MPNN (max) | No | Accuracy | 97.30% |
| MNIST Superpixels | PNA (no scalers) | No | Accuracy | 97.25% |
| MNIST Superpixels | PNA | No | Accuracy | 96.99% |
| MNIST Superpixels | MPNN (sum) | Yes | Accuracy | 95.97% |
| MNIST Superpixels | MPNN (max) | Yes | Accuracy | 97.67% |
| MNIST Superpixels | PNA (no scalers) | Yes | Accuracy | 97.73% |
| MNIST Superpixels | PNA | Yes | Accuracy | 97.57% |
| MolHIV | MPNN (sum) | No | ROC-AUC | 76.08% |
| MolHIV | MPNN (max) | No | ROC-AUC | 76.48% |
| MolHIV | PNA (no scalers) | No | ROC-AUC | 77.32% |
| MolHIV | PNA | No | ROC-AUC | 77.23% |

PNA with scalers achieved the lowest ZINC MAE, particularly when edge features were included. On the regular superpixel graph structures, PNA variants were close to the MPNN-max baseline, and removing scalers was slightly better in some settings. On MolHIV, PNA without scalers gave the strongest ROC-AUC among the tested variants. The report notes that these experiments were run once due to time and compute limits, so the results do not include repeated-run variability.

## Running the notebook

The notebook was prepared for Google Colab/Kaggle with GPU acceleration. It installs PyTorch 2.3.0 with CUDA 12.1 wheels and DGL, then loads datasets through DGL and OGB. Dataset preparation includes mounting Google Drive for the Superpixel archive and MolHIV data, so update those cells to match your own dataset locations and runtime before running. Training is computationally intensive; the benchmark configurations range from 40 to 1,000 epochs.

Run the notebook top to bottom after adapting the installation, accelerator, and data-loading cells to your environment. The notebook includes code for the experiments, but it contains only part of the CIFAR10 and MNIST training results because training was split across two notebook sessions.

## References

- Corso, G. et al. “Principal Neighbourhood Aggregation for Graph Nets.” *NeurIPS*, 2020. [Paper](https://arxiv.org/abs/2004.05718)
- Xu, K. et al. “How Powerful are Graph Neural Networks?” *ICLR*, 2019. [Paper](https://arxiv.org/abs/1810.00826)
- Dwivedi, V. P. et al. “Benchmarking Graph Neural Networks.” *JMLR*, 2023.
- Hu, W. et al. “Open Graph Benchmark: Datasets for Machine Learning on Graphs.” *NeurIPS*, 2020.
