
## Introduction

## Background

## QCANet

## 3D U-net

### Loss

Since the dice loss of the library we used produced negative loss, we decided to derive the same loss from the [paper](https://arxiv.org/pdf/1606.04797.pdf) the library references and use that as a loss function. This is the formula used:

$$D = \frac{2 \times \sum_{i}^{N} p_{i} g_{i}}{\sum_{i}^{N} p_{i}^{2} + \sum_{i}^{N} g_{i}^{2}}$$

where the sums run over the $N$ voxels, of the predicted binary segmentation volume $p_i \in P$ and the ground truth binary volume $g_i \in G$.

## Results

## Discussion

## Conclusion
