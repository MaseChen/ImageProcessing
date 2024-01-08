import cv2
import numpy as np
import copy
from cv2.typing import MatLike


# 指数灰度变换
def exp_gray_transform_algorithm(
    image: MatLike, c: float = 0.00000005, v: float = 4.0
) -> np.uint8:
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i**v
    result = cv2.LUT(image, lut)
    result = np.uint8(result + 0.5)  # type:ignore
    return result


# 伽马校正
def gamma_correction_algorithm(image: MatLike, gamma: float = 0.5) -> MatLike:
    invgamma = 1 / gamma
    result = np.array(
        np.power((image / 255), invgamma) * 255, dtype=np.uint8  # type: ignore
    )
    return result


# 彩色负片
def color_negative_algorithm(image: MatLike) -> MatLike:
    result = cv2.bitwise_not(image)
    return result


# 拉普拉斯锐化
def lapras_algorithm(image: MatLike) -> MatLike:
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)  # type: ignore
    result = cv2.convertScaleAbs(dst)
    return result


# 傅里叶变换
def fourier_transform_algorithm(image: MatLike) -> np.uint8:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)  # type: ignore
    dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)
    magnitude_v = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    result = np.uint8(cv2.cvtColor(magnitude_v, cv2.COLOR_GRAY2BGR))  # type: ignore
    return result


# 逆滤波复原
def inverse_filter_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)  # type: ignore
    dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape

    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 20 : crow + 20, ccol - 20 : ccol + 20] = 1

    inverse = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            H = mask[i, j]
            inverse[i, j] = 1 / H
    fshift = dft_shift * inverse
    ishift = np.fft.ifftshift(fshift)
    result = cv2.idft(ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])  # type: ignore
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# 维纳滤波复原
def wiener_filter_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)  # type: ignore
    dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape

    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 20 : crow + 20, ccol - 20 : ccol + 20] = 1

    wiener = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            H = mask[i, j]
            wiener[i, j] = (abs(H) ** 2) / (H * (abs(H) ** 2 + 0.0025))
    fshift = dft_shift * wiener
    ishift = np.fft.ifftshift(fshift)
    result = cv2.idft(ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# 均值滤波
def mean_filter_algorithm(image: MatLike) -> MatLike:
    result = cv2.blur(image, (3, 3))
    return result


# 中值滤波
def median_filter_algorithm(image: MatLike) -> MatLike:
    result = cv2.medianBlur(image, 3)
    return result


# 高斯低通滤波
def lowpass_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)  # type: ignore
    dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 20 : crow + 20, ccol - 20 : ccol + 20] = 1
    fshift = dft_shift * mask
    ishift = np.fft.ifftshift(fshift)
    result = cv2.idft(ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# 高斯高通滤波
def highpass_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float32 = np.float32(gray)  # type: ignore
    dft = cv2.dft(gray_float32, flags=cv2.DFT_COMPLEX_OUTPUT)  # type: ignore
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 20 : crow + 20, ccol - 20 : ccol + 20] = 0
    fshift = dft_shift * mask
    ishift = np.fft.ifftshift(fshift)
    result = cv2.idft(ishift)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# Canny 算法
def canny_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Canny(gray, 100, 200)
    return result


# 外轮廓检测
def contour_detection_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(copy.deepcopy(image), contours, -1, (0, 255, 0), 1)
    return result


# 填充轮廓
def fill_contour_algorithm(result: MatLike) -> MatLike:
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(result, [contour], 0, (0, 255, 0), -1)
    return result


# 全局直方图均衡化
def global_histogram_equalization_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(gray)
    return result


# 局部直方图均衡化
def local_histogram_equalization_algorithm(image: MatLike) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    return result


# 限制对比度自适应直方图均衡化
def contrast_limited_adaptive_histogram_equalization_algorithm(
    image: MatLike,
) -> MatLike:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_equalized = clahe.apply(L)
    lab = cv2.merge((L_equalized, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return result
