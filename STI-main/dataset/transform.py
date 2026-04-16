import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
from utils import *
import cv2
import random
import numpy as np
import cv2
from PIL import Image
import torch

def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def normalize_back(img, mask=None, dataset='pascal'):
    means = np.asarray([0.485, 0.456, 0.406])
    stds = np.asarray([0.229, 0.224, 0.225])
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img*stds + means
    img = img*255

    if mask is not None:
        cmap = color_map(dataset)

        mask = mask.squeeze(0).cpu().numpy()
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.putpalette(cmap)
        return img, mask

    return img


def resize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask



def cutin(img, mask, p=1):
    if random.random() < p:
        # 获取图像尺寸
        size = img.size()
        W = size[1]
        H = size[2]

        # 根据 lambda 值计算剪裁尺寸
        lam = random.uniform(0.2, 0.7)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # 使用基于深度学习的显著性检测器计算显著性地图
        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        temp_mask = mask.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.ObjectnessBING_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        # 找到显著性地图中最大值的索引，以确定感兴趣区域的中心
        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        x = maximum_indices[0]
        y = maximum_indices[1]

        # 根据剪裁尺寸计算边界框坐标
        bbx1 = np.clip(x - cut_w // 2, 0, W)
        bby1 = np.clip(y - cut_h // 2, 0, H)
        bbx2 = np.clip(x + cut_w // 2, 0, W)
        bby2 = np.clip(y + cut_h // 2, 0, H)

        # 调整图像大小以适应感兴趣区域
        roi_width = bbx2 - bbx1
        roi_height = bby2 - bby1
        fx = roi_width / size[1]
        fy = roi_height / size[2]

        # 缩小图像并调整大小以适应感兴趣区域
        resized_image = cv2.resize(temp_img, (0, 0), fx=fx, fy=fy)
        resized_mask = cv2.resize(temp_mask, (0, 0), fx=fx, fy=fy)

        # 随机旋转图像和掩码
        angle = random.randint(0, 360)  # 随机选择一个角度
        M = cv2.getRotationMatrix2D((resized_image.shape[1] / 2, resized_image.shape[0] / 2), angle, 1)
        rotated_image = cv2.warpAffine(resized_image, M, (resized_image.shape[1], resized_image.shape[0]))
        rotated_mask = cv2.warpAffine(resized_mask, M, (resized_mask.shape[1], resized_mask.shape[0]))

        # 将旋转后的感兴趣区域复制到原始图像和掩码中
        img[:, bbx1:bbx2, bby1:bby2] = torch.tensor(rotated_image).permute(2, 0, 1)
        mask[:, bbx1:bbx2, bby1:bby2] = torch.tensor(rotated_mask).permute(2, 0, 1)

        # 随机水平翻转图像和掩码
        if random.random() < 1:
            img = img.flip(-1)
            mask = mask.flip(-1)

        # 随机垂直翻转图像和掩码
        if random.random() < 1:
            img = img.flip(-2)
            mask = mask.flip(-2)

    # 将处理后的 img 和 mask 转换为 PIL 图像对象
    img = Image.fromarray(img.numpy().astype(np.uint8).transpose(1, 2, 0))
    mask = Image.fromarray(mask.numpy().astype(np.uint8).transpose(1, 2, 0))

    return img, mask