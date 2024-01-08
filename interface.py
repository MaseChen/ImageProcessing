import cv2
import os
import glob
import time
from typing import Optional
from cv2.typing import MatLike
from PIL import Image, ImageTk
from tkinter import (
    filedialog,
    messagebox,
    Tk,
    Menu,
    Canvas,
    LEFT,
    NW,
)
import algorithms as alg


# 全局变量
image_paths: Optional[list[str]] = None
image_paths_index: int = 0
image_path: Optional[str] = None
image: Optional[MatLike] = None
modified_image: Optional[MatLike] = None
canvas_for_show: Optional[Canvas] = None
image_on_canvas: Optional[int] = None
history: list[MatLike] = []
history_index: int = -1


# 初始化显示图片
def init_image() -> None:
    global image_paths, image_paths_index, image_path, image, modified_image, history, history_index
    image_paths = list()
    image_paths_index = 0
    for image_path in glob.glob(os.getcwd() + "\\*.png"):
        image_paths.append(image_path)

    for image_path in glob.glob(os.getcwd() + "\\*.jpg"):
        image_paths.append(image_path)

    for image_path in glob.glob(os.getcwd() + "\\*.bmp"):
        image_paths.append(image_path)

    if image_paths:
        image_path = image_paths[image_paths_index]
    if image_path:
        root.title(image_path)
        image = cv2.imread(image_path)
        modified_image = image.copy()
        show_image(image)
        history = [modified_image]
        history_index = 0


# 显示图片
def show_image(img: Optional[MatLike]) -> None:
    global canvas_for_show, image_on_canvas
    if img is None:
        return

    if canvas_for_show is None:
        canvas_for_show = Canvas(root, width=window_width, height=window_height - 50)
        canvas_for_show.pack(side=LEFT, padx=10, pady=10)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img)
    img2 = ImageTk.PhotoImage(img1)
    image_on_canvas = canvas_for_show.create_image(0, 0, anchor=NW, image=img2)
    canvas_for_show.image = img2  # type: ignore


# 左方向键
def left_key_callback(_) -> None:
    global image_paths, image_paths_index, image_path, image, modified_image, history, history_index
    if image_paths is None or len(image_paths) == 0:
        return
    if image_paths_index > 0:
        image_paths_index = image_paths_index - 1
    else:
        image_paths_index = len(image_paths) - 1
    image_path = image_paths[image_paths_index]
    if image_path:
        root.title(image_path)
        image = cv2.imread(image_path)
        modified_image = image.copy()
        show_image(image)
        history = [modified_image]
        history_index = 0


# 右方向键
def right_key_callback(_) -> None:
    global image_paths, image_paths_index, image_path, image, modified_image, history, history_index
    if image_paths is None or len(image_paths) == 0:
        return
    image_paths_index = (image_paths_index + 1) % len(image_paths)
    image_path = image_paths[image_paths_index]
    if image_path:
        root.title(image_path)
        image = cv2.imread(image_path)
        modified_image = image.copy()
        show_image(image)
        history = [modified_image]
        history_index = 0


# 是否保存原有图片
def image_is_saved() -> None:
    if image_path:
        answer = messagebox.askyesno(title="看图软件", message="是否保存当前图片?")
        if answer:
            save_image()
    new_image()


# 新建
def new_image() -> None:
    global image_path, image, modified_image, history, history_index
    root.title("新建图像")
    modified_image = cv2.imread("white.jpg")
    image_path = None
    show_image(modified_image)
    history = [modified_image]
    history_index = 0


# 选择图片
def select_image() -> None:
    global image_paths, image_paths_index, image_path, image, modified_image, history, history_index

    filetypes = (
        ("JPEG Files", "*.jpg"),
        ("PNG Files", "*.png"),
        ("BMP Files", "*.bmp"),
    )
    image_path = filedialog.askopenfilename(
        title="Select Image File", filetypes=filetypes
    )

    folder_part = os.path.normpath(os.path.dirname(image_path))

    image_paths = list()
    image_paths_index = 0
    png_files = glob.glob(os.path.join(folder_part, "*.png"))
    jpg_files = glob.glob(os.path.join(folder_part, "*.jpg"))
    bmp_files = glob.glob(os.path.join(folder_part, "*.bmp"))
    image_paths.extend(png_files)
    image_paths.extend(jpg_files)
    image_paths.extend(bmp_files)

    if image_path:
        root.title(image_path)
        image = cv2.imread(image_path)
        modified_image = image.copy()
        show_image(image)
        history = [modified_image]
        history_index = 0


# 保存图像
def save_image() -> None:
    global modified_image
    if modified_image is None:
        return
    filetypes = (
        ("JPEG Files", "*.jpg"),
        ("PNG Files", "*.png"),
        ("BMP Files", "*.bmp"),
    )
    save_path = filedialog.asksaveasfilename(filetypes=filetypes)
    if save_path:
        cv2.imwrite(save_path, modified_image)


# 撤销
def undo() -> None:
    global modified_image, history_index
    if history_index > 0:
        history_index -= 1
        modified_image = history[history_index]
        show_image(modified_image)


# 重做
def redo() -> None:
    global modified_image, history_index
    if history_index < len(history) - 1:
        history_index += 1
        modified_image = history[history_index]
        show_image(modified_image)


# 指数灰度变换
def exp_gray_transform() -> None:
    global modified_image, history, history_index
    modified_image = alg.exp_gray_transform_algorithm(modified_image)  # type: ignore
    if modified_image is None:
        return
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 彩色负片
def color_negative() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.color_negative_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 均值滤波
def mean_filter() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.mean_filter_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 中值滤波
def median_filter() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.median_filter_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 拉普拉斯锐化
def lapras() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.lapras_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 伽马校正
def gamma_correction_0d4() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 0.4)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


def gamma_correction_0d6() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 0.6)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


def gamma_correction_0d8() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 0.8)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


def gamma_correction_1d2() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 1.2)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


def gamma_correction_1d4() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 1.4)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


def gamma_correction_1d6() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.gamma_correction_algorithm(modified_image, 1.6)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# Canny 算法
def canny() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.canny_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 外轮廓检测
def contour_detection() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.contour_detection_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 填充轮廓
def fill_contour() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.fill_contour_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 全局直方图均衡化
def global_histogram_equalization() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.global_histogram_equalization_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 局部直方图均衡化
def local_histogram_equalization() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.local_histogram_equalization_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 限制对比度自适应直方图均衡化
def contrast_limited_adaptive_histogram_equalization() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.contrast_limited_adaptive_histogram_equalization_algorithm(
        modified_image
    )
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 傅里叶变换
def fourier_transform() -> None:
    global modified_image, history, history_index
    modified_image = alg.fourier_transform_algorithm(modified_image)  # type: ignore
    if modified_image is None:
        return
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 低通滤波
def lowpass() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.lowpass_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 高通滤波
def highpass() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.highpass_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 逆滤波复原
def inverse_filter() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.inverse_filter_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 维纳滤波复原
def wiener_filter() -> None:
    global modified_image, history, history_index
    if modified_image is None:
        return
    modified_image = alg.wiener_filter_algorithm(modified_image)
    show_image(modified_image)
    history.append(modified_image)
    history_index += 1


# 检查升级
def update() -> None:
    time.sleep(1)
    messagebox.showinfo("检查更新", "当前为最新版本")


# 关于
def reference() -> None:
    messagebox.showinfo("关于", "2024 Copyleft cyx and lyt")


root: Tk = Tk()
root.title("看图软件")
menubar: Menu = Menu(root)
root.config(menu=menubar)

window_width: int = 1200
window_height: int = 700
screen_width: int = root.winfo_screenwidth()
screen_height: int = root.winfo_screenheight()
x: int = int((screen_width - window_width) / 2)
y: int = int((screen_height - window_height) / 2)

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.after(10, init_image)
root.bind("<KeyPress-Left>", left_key_callback)
root.bind("<KeyPress-Right>", right_key_callback)

# 文件
file_menu: Menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="新建", command=image_is_saved)
file_menu.add_command(label="打开", command=select_image)
file_menu.add_command(label="保存", command=save_image)
file_menu.add_separator()
file_menu.add_command(label="退出", command=root.quit)
menubar.add_cascade(label="文件", menu=file_menu)

# 编辑
edit_menu: Menu = Menu(menubar, tearoff=0)
edit_menu.add_command(label="撤销", command=undo)
edit_menu.add_command(label="重做", command=redo)
menubar.add_cascade(label="编辑", menu=edit_menu)

# 彩色图像处理
image_menu: Menu = Menu(menubar, tearoff=0)
image_menu.add_command(label="指数灰度变换", command=exp_gray_transform)
image_menu.add_command(label="彩色负片", command=color_negative)
image_menu.add_command(label="均值滤波", command=mean_filter)
image_menu.add_command(label="中值滤波", command=median_filter)
image_menu.add_command(label="拉普拉斯锐化", command=lapras)
menubar.add_cascade(label="彩色图像处理", menu=image_menu)

gamma_menu: Menu = Menu(image_menu, tearoff=0)
gamma_menu.add_command(label="0.4", command=gamma_correction_0d4)
gamma_menu.add_command(label="0.6", command=gamma_correction_0d6)
gamma_menu.add_command(label="0.8", command=gamma_correction_0d8)
gamma_menu.add_command(label="1.2", command=gamma_correction_1d2)
gamma_menu.add_command(label="1.4", command=gamma_correction_1d4)
gamma_menu.add_command(label="1.6", command=gamma_correction_1d6)
image_menu.add_cascade(label="伽马校正", menu=gamma_menu)

# 频率域操作
filter_menu: Menu = Menu(menubar, tearoff=0)
filter_menu.add_command(label="傅里叶变换频谱", command=fourier_transform)
filter_menu.add_command(label="低通滤波", command=lowpass)
filter_menu.add_command(label="高通滤波", command=highpass)
menubar.add_cascade(label="频率域操作", menu=filter_menu)

detection_menu: Menu = Menu(image_menu, tearoff=0)
detection_menu.add_command(label="Canny 算法", command=canny)
detection_menu.add_command(label="外轮廓检测", command=contour_detection)
detection_menu.add_command(label="填充轮廓", command=fill_contour)
image_menu.add_cascade(label="边缘检测", menu=detection_menu)

equalization_menu: Menu = Menu(image_menu, tearoff=0)
equalization_menu.add_command(label="全局直方图均衡化", command=global_histogram_equalization)
equalization_menu.add_command(label="局部直方图均衡化", command=local_histogram_equalization)
equalization_menu.add_command(
    label="限制对比度自适应直方图均衡化", command=contrast_limited_adaptive_histogram_equalization
)
image_menu.add_cascade(label="直方图均衡化", menu=equalization_menu)

# 图像复原
recovery_menu: Menu = Menu(menubar, tearoff=0)
recovery_menu.add_command(label="逆滤波复原", command=inverse_filter)
recovery_menu.add_command(label="维纳滤波复原", command=wiener_filter)
menubar.add_cascade(label="图像复原", menu=recovery_menu)

# 帮助
help_menu: Menu = Menu(menubar, tearoff=0)
help_menu.add_command(label="检查更新", command=update)
help_menu.add_command(label="关于", command=reference)
menubar.add_cascade(label="帮助", menu=help_menu)


root.mainloop()
