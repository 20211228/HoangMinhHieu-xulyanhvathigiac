import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# Hàm tính khoảng cách Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Hàm khởi tạo các trọng tâm ban đầu
def initialize_centroids(data, k):
    random_indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in random_indices]
    return centroids


# Hàm gán mỗi điểm vào cụm gần nhất
def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = distances.index(min(distances))
        clusters[closest_centroid].append(point)
        labels.append(closest_centroid)
    return clusters, labels


# Hàm cập nhật trọng tâm cụm
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Nếu cụm có điểm
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(None)
    return new_centroids


# Hàm chính cho k-means
def k_means(data, k, max_iterations=100, tolerance=1e-4):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters, labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        # Kiểm tra hội tụ
        converged = True
        for old, new in zip(centroids, new_centroids):
            if new is not None and euclidean_distance(old, new) > tolerance:
                converged = False

        centroids = [new if new is not None else old for old, new in zip(centroids, new_centroids)]

        if converged:
            break
    return labels, centroids


# Đọc ảnh và chuyển đổi thành mảng pixel
image = cv2.imread('animals.png')
if image is None:
    print("Không tìm thấy ảnh.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển thành RGB cho matplotlib
    pixels = image_rgb.reshape(-1, 3)  # Chuyển thành mảng 2D, mỗi dòng là một pixel

    # Giá trị k cần phân cụm
    k_values = [2, 3, 4, 5]

    plt.figure(figsize=(15, 10))

    # Hiển thị ảnh gốc
    plt.subplot(2, 3, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image_rgb)
    plt.axis("off")

    # Phân cụm và hiển thị cho từng giá trị k
    for idx, k in enumerate(k_values, 2):  # Bắt đầu từ vị trí thứ 2
        labels, centroids = k_means(pixels, k)

        # Tái tạo ảnh theo cụm
        segmented_image = np.array([centroids[label] for label in labels])
        segmented_image = segmented_image.reshape(image_rgb.shape).astype(np.uint8)

        # Hiển thị ảnh phân cụm
        plt.subplot(2, 3, idx)
        plt.title(f"Ảnh phân cụm (k={k})")
        plt.imshow(segmented_image)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
