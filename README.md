# Principal Neighborhood Aggregation (PNA) Model for Graph-Level Attribute Prediction

This project investigates whether Principal Neighbourhood Aggregation (PNA) improves the expressive power of message-passing graph neural networks on continuous node features and varied graph structures. It compares PNA models with conventional MPNNs on four graph-level prediction benchmarks: ZINC, CIFAR10 Superpixels, MNIST Superpixels, and OGBG-MolHIV.

The implementation and experiment outputs are in [`code.ipynb`](code.ipynb).

## Methods

Let $x_i^{(t)}$ be the node feature of node $i$ at step $t$ and $N(i)$ be the set of all neighbours of node $i$. The updation of node feature at each step is

$$
    x_i^{(t+1)} = \phi (x_i^{(t)}, \Box_{j \in N(i)}  \psi (x_i^{(t)},x_j^{(t)}))
$$

where $\psi$ and $\phi$ are the message function and update function, respectively. The Principal Neighborhood Aggregation combines four neighbourhood aggregators with degree-based amplification and attenuation scalers:

$$
\Box =
\begin{bmatrix}
I \\
S(D, \alpha = 1) \\
S(D, \alpha = -1)
\end{bmatrix}
\bigotimes
\begin{bmatrix}
M_1 \\
M_2 \\
\max \\
\min
\end{bmatrix}
$$

$S(d, \alpha) = (\log(d+1) / \delta)^{\alpha}$ is a degree-based scaler, where $d$ is the degree of the node receiving the message, and $\delta = \sum_{i \in V_\text{train}}\log(d_i+1)/|V_\text{train}|$.

<img width="2997" height="1000" alt="Fig_2" src="https://github.com/user-attachments/assets/fd3f9f1e-ee37-492b-ad6e-df3a0f84a1aa" />

The experiments compare PNA with scalers, PNA without scalers, and MPNNs using sum or maximum aggregation. For ZINC and the image benchmarks, variants are evaluated with and without edge features where applicable. MolHIV experiments use node features only. For the case of considering edge features, our update equation of node features becomes $x_i^{(t+1)} = \phi (x_i^{(t)}, \Box_{j \in N(i)}  \psi (x_i^{(t)}, e_{ji}, x_j^{(t)}))$.


## Datasets

In ZINC dataset, the graphs are molecules whose node and edge features are one-hot vectors representing the atom and bond types, respectively. It has 10,000 train, 1,000 validation and 1,000 test graphs. The task is predicting the constrained solubility of molecules. In this dataset, we use the Mean Absolute Error (MAE) as the loss function and evaluation metric.

Both MNIST and CIFAR10 datasets consist of graphs constructed from images across 10 classes using a superpixel representation. CIFAR10 dataset has 45,000 train, 5,000 validation and 10,000 test graphs, while MNIST dataset has 55,000 train, 5,000 validation and 10,000 test graphs. The tasks for these datasets are both to classify the images into 10 classes. We use the cross entropy as the loss function and the accuracy as the evaluation metric.

In the MolHIV dataset, the graphs are molecules whose node features contain atomic number, chirality, and other atom features. The training set, validation set and test set are divided into proportions of 80\%, 10\% and 10\%, respectively. The task is a binary classification of molecular properties and we use binary cross entropy as the loss function and ROC-AUC as the evaluation metric.


## Experimental Results

| Model | ZINC (No edge feature) MAE | ZINC (Edge feature) MAE | CIFAR10 (No edge feature) Acc | CIFAR10 (Edge feature) Acc | MNIST (No edge feature) Acc | MNIST (Edge feature) Acc | MolHIV (No edge feature) ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNN (sum) | 0.3302 | 0.2569 | 61.94 | 63.63 | 96.15 | 95.97 | 76.08 |
| MPNN (max) | 0.4364 | 0.2755 | 68.46 | 69.87 | 97.30 | 97.67 | 76.48 |
| PNA (no sc.) | 0.3945 | 0.2006 | 70.02 | 69.42 | 97.25 | 97.73 | 77.32 |
| **PNA** | **0.2594** | **0.1799** | **69.14** | **68.35** | **96.99** | **97.57** | **77.23** |


Compared to the results on ZINC, the MAE of the PNA models with and without scalers was significantly lower than that of the MPNN (max), while their accuracy on the CIFAR dataset showed little difference. This is because the chemical molecules in the ZINC dataset have more complex adjacency relationships, and edge features (the type of chemical bond) can influence the properties of the molecules. In contrast, image data consists of regular grids and the edge features are not important. In such simple graph structures, the maximum aggregator may be sufficient.


## References

- Corso, G. et al. “Principal Neighbourhood Aggregation for Graph Nets.” *NeurIPS*, 2020. [Paper](https://arxiv.org/abs/2004.05718)
- Xu, K. et al. “How Powerful are Graph Neural Networks?” *ICLR*, 2019. [Paper](https://arxiv.org/abs/1810.00826)
- Dwivedi, V. P. et al. “Benchmarking Graph Neural Networks.” *JMLR*, 2023.
- Hu, W. et al. “Open Graph Benchmark: Datasets for Machine Learning on Graphs.” *NeurIPS*, 2020.
