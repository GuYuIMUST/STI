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

