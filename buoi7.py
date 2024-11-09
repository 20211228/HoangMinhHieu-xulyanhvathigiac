import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score
import skfuzzy as fuzz
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import LabelEncoder
import cv2  # OpenCV để đọc và xử lý ảnh

# -----------------------------
# Phân cụm trên bộ dữ liệu IRIS
# -----------------------------
# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('Iris2/Iris.csv')  # Thay đường dẫn đúng tới tệp CSV

# Lấy các đặc trưng (features) và nhãn (labels)
X_iris = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y_iris = df['Species'].values

# -----------------------------
# Mã hóa nhãn (labels) thành số
# -----------------------------
le = LabelEncoder()
y_iris_encoded = le.fit_transform(y_iris)

# -----------------------------
# Phân cụm với Fuzzy C-means (FCM)
# -----------------------------
n_clusters = len(np.unique(y_iris))  # Số cụm = số lớp trong nhãn
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_iris.T, n_clusters, 2, error=0.005, maxiter=1000)

# Gán nhãn cho mỗi điểm dữ liệu theo cụm có độ tin cậy cao nhất
y_fcm = np.argmax(u, axis=0)

# -----------------------------
# Phân cụm với K-means
# -----------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X_iris)

# -----------------------------
# Phân cụm với AHC
# -----------------------------
Z = linkage(X_iris, method='ward')
y_ahc = fcluster(Z, t=n_clusters, criterion='maxclust')

# -----------------------------
# Đánh giá kết quả phân cụm
# -----------------------------
# F1-score và Rand Index
f1_kmeans = f1_score(y_iris_encoded, y_kmeans, average='weighted')
f1_fcm = f1_score(y_iris_encoded, y_fcm, average='weighted')
f1_ahc = f1_score(y_iris_encoded, y_ahc, average='weighted')

rand_kmeans = adjusted_rand_score(y_iris_encoded, y_kmeans)
rand_fcm = adjusted_rand_score(y_iris_encoded, y_fcm)
rand_ahc = adjusted_rand_score(y_iris_encoded, y_ahc)

# In kết quả
print("Kết quả phân cụm trên bộ dữ liệu IRIS:")
print(f"F1-score K-means: {f1_kmeans}")
print(f"F1-score FCM: {f1_fcm}")
print(f"F1-score AHC: {f1_ahc}")

print(f"Rand Index K-means: {rand_kmeans}")
print(f"Rand Index FCM: {rand_fcm}")
print(f"Rand Index AHC: {rand_ahc}")


# -----------------------------
# Phân cụm trên ảnh vệ tinh giao thông
# -----------------------------
# Đọc ảnh vệ tinh (Đảm bảo bạn có tệp ảnh này)
image = cv2.imread("ggmap2.png")  # Đảm bảo rằng bạn đã có file ảnh này
if image is None:
    print("Không thể mở ảnh, kiểm tra lại đường dẫn hoặc định dạng ảnh!")
    exit()

# Giảm độ phân giải của ảnh để giảm kích thước dữ liệu
image_resized = cv2.resize(image, (100, 100))  # Giảm độ phân giải xuống còn 100x100

# Chuyển ảnh thành dạng 2D với mỗi pixel là một điểm (mỗi pixel có 3 kênh màu RGB)
pixels = image_resized.reshape((-1, 3))

# Sử dụng kiểu dữ liệu float32 thay vì float64
pixels = pixels.astype(np.float32)

# -----------------------------
# Phân cụm với Fuzzy C-means (FCM) cho ảnh vệ tinh
# -----------------------------
n_clusters = 3  # Số cụm
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pixels.T, n_clusters, 2, error=0.005, maxiter=1000)

# Gán nhãn cho mỗi pixel theo cụm có độ tin cậy cao nhất
y_fcm_image = np.argmax(u, axis=0)

# -----------------------------
# Phân cụm với K-means cho ảnh vệ tinh
# -----------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans_image = kmeans.fit_predict(pixels)

# -----------------------------
# Phân cụm với AHC cho ảnh vệ tinh
# -----------------------------
Z = linkage(pixels, method='ward')
y_ahc_image = fcluster(Z, t=n_clusters, criterion='maxclust')

# Vì không có nhãn thực tế, ta sẽ tính Rand Index cho ảnh vệ tinh
rand_kmeans_image = adjusted_rand_score(y_kmeans_image, y_kmeans_image)  # Giả sử không có nhãn thực tế
rand_fcm_image = adjusted_rand_score(y_fcm_image, y_fcm_image)
rand_ahc_image = adjusted_rand_score(y_ahc_image, y_ahc_image)

# In kết quả phân cụm trên ảnh vệ tinh
print("\nKết quả phân cụm trên ảnh vệ tinh:")
print(f"Rand Index K-means (image): {rand_kmeans_image}")
print(f"Rand Index FCM (image): {rand_fcm_image}")
print(f"Rand Index AHC (image): {rand_ahc_image}")
