# STI
## Introduction
**Enhancing Semi-Supervised Semantic Segmentation via Image Search and Advanced Pooling Strategies** 

Sorry, the paper is currently under review. Once the paper is published, we will release the code immediately.
## Results

### STI

ResNet50 and DeepLabv3+                       
| Method              |mIoU-1/16(%) |mIoU-1/8(%)    |mIoU-1/4(%)     |
| ------------------- | ----------- | ------------- | -------------- |
| SupOnly             | 62.99       | 65.08         | 68.85          |
| ESC                 | -           | 70.2          | 72.6           |
| DCC                 | 70.1        | 72.4          | 74.0           |
| MT                  | 66.77		    | 70.78         | 73.22          |
| CCT                 | 65.22		    | 70.87         | 73.43          |
| GCT                 | 64.05		    | 70.47         | 73.45          |
| CPS                 | 68.21		    | 73.20         | 74.24          |
| CTT                 | -		        | 73.66         | 75.07          |
| ELN                 | -		        | 70.34         | 73.52          |
| USCS                | 72.30       | 74.88         | 76.15          |
| RRN                 | 73.38		    | 74.91         | 76.80          |
| PGCL                | -		        | 75.20         | 76.00          |
| CPCL                | 71.66		    | 73.74         | 75.35          |
| FPL                 | 72.52		    | 73.74         | 75.35          |
| RWMS                | 72.20		    | 75.03         | 76.63          |
| ST++                | 72.6        | 74.4          | 75.4           |
|**STI(ours)**        | **73.75**   | **75.98**     | **76.14**      |

ResNet101 and DeepLabv3+
| Method              |mIoU-1/16(%) |mIoU-1/8(%)    |mIoU-1/4(%)     |
| ------------------- | ----------- | ------------- | -------------- |
| SupOnly             | 64.97       | 67.57         | 70.45          |
| AdvSeg              | 68.2		    | 69.5          | -              |
| MT                  | 69.8		    | 71.5          | 73.0           |
| S4GAN               | 69.1		    | 72.4          | 74.5           |
| GCT                 | 67.2		    | 72.5          | 75.1           |
| CCT                 | 70.8		    | 72.2          | 75.1           |
| PseudoSeg           | -		        | 73.2          | -              |
| DCC                 | 72.4		    | 74.6          | 76.3           |
| PC2Seg              | -		        | 74.1          | -              |
| CPS                 | 69.8			  | 74.3          | 74.6           |
| AEL                 | 74.5			  | 75.6          | 77.5           |
| CutMix              | 67.98		    | 69.15         | 73.66          |
| U2PL                | 74.9		    | 76.5          | 78.5           |
| ST                  | 72.9		    | 75.7          | 76.4           |
| ST++                | 74.5        | 76.3          | 76.6           |
|**STI(ours)**        | **75.51**   | **76.90**     | **76.98**      |



## Usage
To run our code, you may need one GeForce RTX 3090(24G memory).

#### Train and Eval
```bash 
python train.py
python eval.py
```

### Requirements

To ensure the code can run, we provide versions of some libraries.

- apex-0.1
- python-3.8.13
- numpy-1.23.2
- torch-1.8.1
- pandas-1.5.3
- opencv-python-4.8.1

## Acknowledgement 

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

 **This code is based on the implementation of  [ST++](https://github.com/quark0/darts), [CISC-R](https://github.com/xiaomi-automl/FairDARTS), [Cutout](https://github.com/uoguelph-mlrg/Cutout) and [SoftPool](https://github.com/alexandrosstergiou/SoftPool).**

## Selected References

If there are any missing citations, please contact us. It is an unintentional omission, and we will add the citations accordingly.

- Yang L, Zhuo W, Qi L, Shi Y, Gao Y.: St++: Make self-training work better for semi-supervised semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4268-4277 (2022).
- Wu L, Fang L, He X, He M, Ma J, Zhong Z.: Querying Labeled for Unlabeled: Cross-Image Semantic Consistency Guided Semi-Supervised Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):8827-8844 (2023).
- DeVries, Terrance. "Improved Regularization of Convolutional Neural Networks with Cutout." arxiv preprint arxiv:1708.04552 (2017).
- Stergiou, A., Poppe, R., & Kalliatakis, G. Refining activation downsampling with SoftPool. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10357-10366 (2021).

