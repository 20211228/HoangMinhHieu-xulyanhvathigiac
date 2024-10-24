import cv2
import numpy as np
from matplotlib import pyplot as plt

# Hàm chuyển đổi ảnh sang ảnh âm tính
def negative_image(image):
    intensity_max = 255
    return intensity_max - image

# Hàm tăng độ tương phản của ảnh
def increase_contrast(image, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Hàm biến đổi log để tăng cường chi tiết nhỏ trong ảnh
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image[np.isinf(log_image)] = 0  # Xử lý giá trị vô cực
    log_image[np.isnan(log_image)] = 0  # Xử lý giá trị không phải số
    return np.array(log_image, dtype=np.uint8)

# Hàm cân bằng histogram để cải thiện độ tương phản của ảnh
def histogram_equalization(image):
    if len(image.shape) == 2:  # Ảnh grayscale
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Ảnh màu
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Hàm vẽ lược đồ xám
def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

# Hàm xử lý ảnh đầu vào
def process_image(image_path):
    # Đọc ảnh đầu vào và chuyển về ảnh xám 8-bit
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Vẽ lược đồ xám ban đầu
    plot_histogram(image, 'Original Grayscale Histogram')

    # Áp dụng các phép biến đổi ảnh
    negative = negative_image(image)
    contrast = increase_contrast(image)
    log_transformed = log_transform(image)
    hist_equalized = histogram_equalization(image)

    # Vẽ lược đồ xám sau khi cân bằng
    plot_histogram(hist_equalized, 'Equalized Grayscale Histogram')

    return negative, contrast, log_transformed, hist_equalized

# Hàm hiển thị các ảnh đã xử lý
def display_images(images, titles):
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # Đường dẫn đến ảnh đầu vào
    image_path = './phongcanh.png'

    # Xử lý ảnh và nhận kết quả
    negative, contrast, log_transformed, hist_equalized = process_image(image_path)

    # Danh sách các ảnh và tiêu đề tương ứng
    images = [negative, contrast, log_transformed, hist_equalized]
    titles = ['Negative Image', 'Increased Contrast', 'Log Transform', 'Histogram Equalization']

    # Hiển thị các ảnh đã xử lý
    display_images(images, titles)