import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh vệ tinh
image = cv2.imread('ggmap3.png', cv2.IMREAD_GRAYSCALE)

# Bước 1: Làm mờ ảnh bằng bộ lọc Gaussian để giảm nhiễu
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Bước 2: Áp dụng các toán tử phát hiện cạnh

# 2.1 Toán tử Sobel
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# 2.2 Toán tử Prewitt
prewittx = cv2.filter2D(blurred, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitty = cv2.filter2D(blurred, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))

# 2.3 Toán tử Roberts
robertsx = cv2.filter2D(blurred, -1, np.array([[1, 0], [0, -1]]))
robertsy = cv2.filter2D(blurred, -1, np.array([[0, 1], [-1, 0]]))
roberts = cv2.magnitude(robertsx.astype(float), robertsy.astype(float))

# 2.4 Toán tử Canny
canny = cv2.Canny(blurred, 100, 200)

# Bước 3: Hiển thị các kết quả phân đoạn cạnh

# Khởi tạo lưới hiển thị 2 hàng x 3 cột với khoảng cách rộng hơn
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

# Tăng khoảng cách giữa các ảnh và tiêu đề
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Hiển thị ảnh gốc
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Ảnh gốc', fontsize=12)
axes[0].axis('off')

# Hiển thị ảnh sau khi dùng Sobel
axes[1].imshow(sobel, cmap='gray')
axes[1].set_title('Sobel', fontsize=12)
axes[1].axis('off')

# Hiển thị ảnh sau khi dùng Prewitt
axes[2].imshow(prewitt, cmap='gray')
axes[2].set_title('Prewitt', fontsize=12)
axes[2].axis('off')

# Hiển thị ảnh sau khi dùng Roberts
axes[3].imshow(roberts, cmap='gray')
axes[3].set_title('Roberts', fontsize=12)
axes[3].axis('off')

# Hiển thị ảnh sau khi dùng Canny
axes[4].imshow(canny, cmap='gray')
axes[4].set_title('Canny', fontsize=12)
axes[4].axis('off')

# Hiển thị ảnh đã làm mờ bằng Gaussian
axes[5].imshow(blurred, cmap='gray')
axes[5].set_title('Gaussian Blurred', fontsize=12)
axes[5].axis('off')

plt.show()
