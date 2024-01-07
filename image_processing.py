import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np
import copy

# 定义全局变量
drawing = False
ix, iy = -1, -1

# 指数灰度变换


def exp_gray_transform_algorithm(image,c=0.00000005, v=4.0):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(image, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)  
    return output_img
    # # 彩色图像指数灰度变换
    # # 将图像转换为YCrCb空间
    # ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # # 拆分通道
    # Y, Cr, Cb = cv2.split(ycrcb)
    # # 对Y通道进行指数灰度变换
    # Y = c * np.power(Y, v)
    # # 合并通道
    # ycrcb = cv2.merge((Y, Cr, Cb))
    # # 将图像转换回BGR空间
    # result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # # 返回指数灰度变换后的图像
    # return result



    # #灰度化处理：此灰度化处理用于图像二值化
    # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # #伽马变换
    # gamma=copy.deepcopy(gray)
    # rows=image.shape[0]
    # cols=image.shape[1]
    # for i in range(rows):
    #     for j in range(cols):
    #         gamma[i][j]=3*pow(gamma[i][j],0.8)

    # return gamma

    # lut = np.zeros(256, dtype=np.float32)
    # for i in range(256):
    #     lut[i] = c * i ** v
    # output_img = cv2.LUT(image, lut) #像素灰度值的映射
    # # output_img = np.uint8(output_img+0.5)  
    # return output_img

# 伽马校正


def gamma_correction_algorithm(image,gamma=0.5):
    invgamma = 1/gamma
    brighter_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
    return brighter_image
    # # 功能：伽马校正
    # # 将图像转换为YCrCb空间
    # ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # # 拆分通道
    # Y, Cr, Cb = cv2.split(ycrcb)
    # # 对Y通道进行伽马校正
    # # Y = np.power(Y, 0.5)
    # # 合并通道
    # ycrcb = cv2.merge((Y, Cr, Cb))
    # # 将图像转换回BGR空间
    # result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # # 返回伽马校正后的图像
    # return result

# 彩色负片


def color_negative_algorithm(image):
    dst = cv2.bitwise_not(image)
    # height = image.shape[0]
    # width = image.shape[1]
    # channels = image.shape[2]
    # #彩色图像的通道一般有三个，为RGB图像
    # for row in range(height):
    #     for col in range(width):
    #         for c in range(channels):
    #             pv = image[row,col,c]
    #             image[row,col,c] = 255-pv #进行反向修改，会使图片变成负片
    
    return dst

# 拉普拉斯锐化


def lapras_algorithm(image):
    #灰度化处理图像
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #阈值处理
    # ret, grayImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
 
    #拉普拉斯算法
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian

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
    # # (1) 另一种算法
    # #灰度化处理图像
    # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # #阈值处理
    # ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # #Canny算子
    # gaussianBlur = cv2.GaussianBlur(binary, (3,3), 0) #高斯滤波
    # Canny = cv2.Canny(gaussianBlur , 50, 150)
    # return Canny

    # (2) Copilot算法
    # # 功能：空间域彩色图像边缘提取
    # # 将图像转换为YCrCb空间
    # ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # # 拆分通道
    # Y, Cr, Cb = cv2.split(ycrcb)
    # # 对Y通道进行边缘提取
    # Y = cv2.Canny(Y, 100, 200)
    # # 合并通道
    # ycrcb = cv2.merge((Y, Cr, Cb))
    # # 将图像转换回BGR空间
    # result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # # 返回边缘提取后的图像
    # return result

    # (3) 原版算法
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Canny 算法进行边缘检测
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 外轮廓检测


def contour_detection_algorithm(image):
    # # (1) 原算法
    # # 转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 使用二值化处理
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # # 查找轮廓
    # contours, _ = cv2.findContours(
    #     thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 在原图像上绘制轮廓
    # result = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    # return result

    # # (2) 另一个算法
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图像上绘制轮廓
    result = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    return result

# 填充轮廓


def fill_contour_algorithm(image):
    # # (1) 原算法
    # # 转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 使用二值化处理
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # # 查找轮廓
    # contours, _ = cv2.findContours(
    #     thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # 对每个轮廓进行填充
    # for contour in contours:
    #     cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)
    # return image

    # (2) 另一个算法
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用二值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


# # 空间域彩色图像指数灰度变换
# def spatial_domain_color_image_exponential_gray_transform_algorithm(image):
#     # 功能：空间域彩色图像指数灰度变换
#     # 将图像转换为YCrCb空间
#     ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     # 拆分通道
#     Y, Cr, Cb = cv2.split(ycrcb)
#     # 对Y通道进行指数灰度变换
#     Y = np.power(Y, 0.5)
#     # 合并通道
#     ycrcb = cv2.merge((Y, Cr, Cb))
#     # 将图像转换回BGR空间
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     # 返回指数灰度变换后的图像
#     return result

# # 空间域彩色图像彩色负片
# def spatial_domain_color_image_color_negative_algorithm(image):
#     # 功能：空间域彩色图像彩色负片
#     # 将图像转换为YCrCb空间
#     ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     # 拆分通道
#     Y, Cr, Cb = cv2.split(ycrcb)
#     # 对Y通道进行彩色负片变换
#     Y = 255 - Y
#     # 合并通道
#     ycrcb = cv2.merge((Y, Cr, Cb))
#     # 将图像转换回BGR空间
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     # 返回彩色负片变换后的图像
#     return result

# # 空间域彩色图像拉普拉斯锐化
# def spatial_domain_color_image_lapras_algorithm(image):
#     # 功能：空间域彩色图像拉普拉斯锐化
#     # 将图像转换为YCrCb空间
#     ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     # 拆分通道
#     Y, Cr, Cb = cv2.split(ycrcb)
#     # 对Y通道进行拉普拉斯锐化变换
#     Y = cv2.Laplacian(Y, cv2.CV_64F)
#     # 合并通道
#     ycrcb = cv2.merge((Y, Cr, Cb))
#     # 将图像转换回BGR空间
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     # 返回拉普拉斯锐化变换后的图像
#     return result

# # 空间域彩色图像边缘提取
# def spatial_domain_color_image_edge_extraction_algorithm(image):
#     # 功能：空间域彩色图像边缘提取
#     # 将图像转换为YCrCb空间
#     ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     # 拆分通道
#     Y, Cr, Cb = cv2.split(ycrcb)
#     # 对Y通道进行边缘提取
#     Y = cv2.Canny(Y, 100, 200)
#     # 合并通道
#     ycrcb = cv2.merge((Y, Cr, Cb))
#     # 将图像转换回BGR空间
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     # 返回边缘提取后的图像
#     return result

# # 伽马校正
# def gamma_correction_algorithm(image):
#     # 功能：伽马校正
#     # 将图像转换为YCrCb空间
#     ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     # 拆分通道
#     Y, Cr, Cb = cv2.split(ycrcb)
#     # 对Y通道进行伽马校正
#     Y = np.power(Y, 0.5)
#     # 合并通道
#     ycrcb = cv2.merge((Y, Cr, Cb))
#     # 将图像转换回BGR空间
#     result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
#     # 返回伽马校正后的图像
#     return result