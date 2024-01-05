import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np

# 定义全局变量
drawing = False
ix, iy = -1, -1

# 指数灰度变换


def exp_gray_transform_algorithm(image):
    pass

# 伽马校正


def gamma_correction_algorithm(image):
    pass

# 彩色负片


def color_negative_algorithm(image):
    pass

# 拉普拉斯锐化


def lapras_algorithm(image):
    pass

# 傅里叶变换


def fourier_transform_algorithm(image):
    pass

# 逆滤波复原


def inverse_filter_algorithm(image):
    pass

# 维纳滤波复原


def wiener_filter_algorithm(image):
    pass

# 均值滤波


def mean_filter_algorithm(image):
    # 使用3x3的核进行均值滤波
    result = cv2.blur(image, (3, 3))
    return result

# 中值滤波


def median_filter_algorithm(image):
    # 使用3x3的大小进行中值滤波
    result = cv2.medianBlur(image, 3)
    return result

# 高斯低通滤波


def gauss_lowpass_algorithm(image):
    pass

# 高斯高通滤波


def gauss_highpass_algorithm(image):
    pass

# 布特沃斯低通滤波


def butworth_lowpass_algorithm(image):
    pass

# 布特沃斯高通滤波


def butworth_highpass_algorithm(image):
    pass


# 边缘检测
# Canny 算法
def canny_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Canny 算法进行边缘检测
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 外轮廓检测


def contour_detection_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图像上绘制轮廓
    result = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return result

# 填充轮廓


def fill_contour_algorithm(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 对每个轮廓进行填充
    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)
    return image

# 直方图均衡化
# 全局直方图均衡化


def global_histogram_equalization_algorithm(image):
    # 功能：全局直方图均衡化
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用全局直方图均衡化
    equalized = cv2.equalizeHist(gray)
    # 返回均衡化后的图像
    return equalized


# 局部直方图均衡化
def local_histogram_equalization_algorithm(image):
    # 功能：局部直方图均衡化
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建CLAHE对象，定义局部直方图均衡化参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用局部直方图均衡化
    equalized = clahe.apply(gray)
    # 返回均衡化后的图像
    return equalized


# 限制对比度自适应直方图均衡化
def contrast_limited_adaptive_histogram_equalization_algorithm(image):
    # 功能：限制对比度自适应直方图均衡化
    # 将图像转换为Lab空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 拆分通道
    L, a, b = cv2.split(lab)
    # 创建CLAHE对象，定义限制对比度自适应直方图均衡化参数
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用限制对比度自适应直方图均衡化
    L_equalized = clahe.apply(L)
    # 合并通道
    lab = cv2.merge((L_equalized, a, b))
    # 将图像转换回BGR空间
    equalized_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    # 返回均衡化后的图像
    return equalized_image
