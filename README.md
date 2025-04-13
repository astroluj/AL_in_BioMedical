# Comparative Analysis of Active Learning Methods in Biomedical Image Classification with MedMNIST Benchmark - _Official Pytorch implementation of the Heliyon, --, 2025_

__*Uijae Lee, Sekjin Hwang, Jinwoo Choi, Joonsoo Choi*__

Official Pytorch implementation for [the paper](__) published on Heliyon titled "_Comparative Analysis of Active Learning Methods in Biomedical Image Classification with MedMNIST Benchmark_".


## Abstract
_Deep learning has shown significant performance in computer vision tasks, setting new benchmarks in domains such as medical imaging. Achieving such performance often requires extensive human-labeled data, thereby increasing costs with task complexity. Active learning has emerged as a solution for mitigating labeling expenses. This approach selects the most informative samples from a large pool of unlabeled data to construct a more compact training dataset. Based on the approach to data selection, the active learning methods are broadly categorized into two, namely, uncertainty-based method and diversity-based method. Uncertainty-based methods select samples with uncertain model predictions from the large pool of unlabeled data. On the other hand, diversity-based methods focus on samples that captures the overall distribution of the unlabeled dataset. Despite their effectiveness, both strategies exhibit inherent limitations, thereby necessitating the need for hybrid methods that combine uncertainty and diversity criteria. Active learning has been acknowledged for its potential with complex datasets such as medical images, and some studies have indeed tested its efficacy using one or two types of medical data. Nonetheless, the research works related to assessing active learning's efficacy across the diverse range of medical imaging data are limited. In this view, this paper presents a comprehensive analysis of the state-of-the-art active learning techniques applied to medical imaging tasks on the MedMNIST v2 dataset._

## Prerequisites:   
- Linux or macOS
- Python 3.10
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.
- pytorch 2.6.0
- cuda 12.0
- Scikit-learn 1.6.1

## Running code
To train the model(s) and evaluate in the paper, run this command:

```train
python main.py --method random --dataset organamnist
```

