# SRLS

Official implementation of **Synergistic Representation Learning via Dual Sample Interactions for Multi-view Clustering**.

> **Paper:** Synergistic representation learning via dual sample interactions for multi-view clustering  
> **Journal:** Pattern Recognition, 177 (2026), 113270  
> **DOI:** 10.1016/j.patcog.2026.113270

## Introduction

SRLS is a deep multi-view clustering method that learns discriminative representations by modeling dual sample interactions:

- **Inter-sample complementarity:** captures complementary information among samples across different views with a multi-view heterogeneous graph.
- **Intra-sample consistency:** reduces feature redundancy and enhances cross-view consistency through feature reconstruction.

The learned representations are finally used for clustering with k-means.

## Framework

<img width="1440" height="630" alt="image" src="https://github.com/user-attachments/assets/c399c824-b4a0-4e08-bce3-267d53f23bc9" />


## Usage

Put the datasets into the `data/` directory, then run:

```bash
python main.py --dataset Scene15
```

You can replace `Scene15` with other supported datasets such as `CUB`, `Yale`, or `Reuters`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{bai2026synergistic,
  title={Synergistic representation learning via dual sample interactions for multi-view clustering},
  author={Bai, Ruina and Huang, Ruizhang and Qin, Yongbin and Xue, Jingjing and Tian, Rujun},
  journal={Pattern Recognition},
  pages={113270},
  year={2026},
  publisher={Elsevier}
}
```
