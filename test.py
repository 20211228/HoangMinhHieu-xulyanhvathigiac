import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# Tải và chuyển đổi ảnh
img_path = 'animals.png'
img = Image.open(img_path)  # Mở ảnh từ đường dẫn
img_array = np.array(img)  # Trả về mảng numpy của ảnh

k_values = [2, 3, 4, 5]  # Các giá trị k
clustered_images = []  # Lưu trữ ảnh đã phân cụm

for k in k_values:
    # Khởi tạo các centroid ngẫu nhiên
    centroids = []
    for _ in range(k):
        centroid = img_array[random.randint(0, img_array.shape[0]-1),
                             random.randint(0, img_array.shape[1]-1), :]
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Thực hiện vài bước chính của thuật toán K-Means
    for iteration in range(10):  # Số lần lặp sẽ chỉ là 10 để đơn giản
        # Gán các pixel vào các cluster dựa trên centroid gần nhất
        clusters = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=int)
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                # Thủ công tính khoảng cách: sqrt((R-r)^2 + (G-g)^2 + (B-b)^2)
                distances = np.array([
                    np.sqrt(np.sum((img_array[i, j] - c) ** 2)) for c in centroids
                ])
                # Thủ công tìm centroid gần nhất
                min_index = np.argmin(distances)
                clusters[i, j] = min_index

        # Cập nhật các centroid dựa trên trung bình của các pixel được gán
        new_centroids = np.zeros((k, img_array.shape[2]))
        for i in range(k):
            cluster_points = img_array[clusters == i]
            if len(cluster_points) > 0:
                # Thủ công tính trung bình
                new_centroid = np.sum(cluster_points, axis=0) / len(cluster_points)
                new_centroids[i] = new_centroid

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    # Tạo lại ảnh sau khi phân cụm
    clustered_img = np.zeros_like(img_array)
    for i in range(k):
        clustered_img[clusters == i] = centroids[i]

    clustered_images.append(clustered_img)  # Lưu lại hình ảnh đã phân cụm

# Hiển thị ảnh gốc và ảnh sau khi phân cụm
plt.figure(figsize=(15, 6))  # Tạo figure để hiển thị ảnh
plt.subplot(1, len(k_values) + 1, 1)
plt.imshow(img_array)  # Hiển thị ảnh gốc
plt.title('Original Image')
plt.axis('off')

for i, (clustered_img, k) in enumerate(zip(clustered_images, k_values)):
    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.imshow(clustered_img)
    plt.title(f'K = {k}')
    plt.axis('off')

plt.show()  # Hiển thị các ảnh