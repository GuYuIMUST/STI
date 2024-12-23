# STI
## Introduction
**Enhancing Semi-Supervised Semantic Segmentation via Image Search and Advanced Pooling Strategies** 

Sorry, the paper is currently under review. Once the paper is published, we will release the code.
## Results

### STI

ResNet50 and DeepLabv3+                       
| Method              |mIoU-1/16(%) |mIoU-1/8(%)    |mIoU-1/4(%)     |
| ------------------- | ----------- | ------------- | -------------- |
| SupOnly             | 62.99       | 65.08         | 68.85          |
| ESC                 | -           | 70.2          | 72.6           |
| DCC                 | 70.1        | 72.4          | 74.0           |
| ST++                | 72.6        | 74.4          | 75.4           |
|**STI(ours)**        | **73.75**   | **75.98**     | **76.14**      |

ResNet101 and DeepLabv3+
| Method              |mIoU-1/16(%) |mIoU-1/8(%)    |mIoU-1/4(%)     |
| ------------------- | ----------- | ------------- | -------------- |
| SupOnly             | 64.97       | 67.57         | 70.45          |
| ESC                 | 69.1        | 72.4          | 74.5           |
| DCC                 | 67.2        | 72.5          | 75.1           |
| ST++                | 74.5        | 76.3          | 76.6           |
|**STI(ours)**        | **75.51**   | **76.90**     | **76.98**      |



## Usage
To run our code, you only need one GeForce RTX 3090(24G memory).

#### Train and Eval
python train.py
python eval.py

### Requirements

To ensure the code can run, we provide versions of some libraries.

- python-3.7.13
- numpy-1.21.5
- pytorch-1.21.1
- pandas-1.3.5
- opencv-python-4.8.1

## Acknowledgement 

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

 **This code is based on the implementation of  [ST++](https://github.com/quark0/darts) and [CISC-R](https://github.com/xiaomi-automl/FairDARTS).**

## Selected References

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

- W. Zhu, C. Liu, W. Fan, X. Xie, Deeplung: Deep 3d dual path nets for automated pulmonary nodule detection and classification,  2018 IEEE winter conference on applications of computer vision (WACV), IEEE2018, pp. 673-681.
- H. Liu, K. Simonyan, Y. Yang, Darts: Differentiable architecture search, arXiv preprint arXiv:1806.09055, (2018).
- Y. Xu, L. Xie, X. Zhang, X. Chen, G.-J. Qi, Q. Tian, H. Xiong, Pc-darts: Partial channel connections for memory-efficient architecture search, arXiv preprint arXiv:1907.05737, (2019).
- H. Jiang, F. Shen, F. Gao, W. Han, Learning efficient, explainable and discriminative representations for pulmonary nodules classification, Pattern Recognition, 113 (2021) 107825.

