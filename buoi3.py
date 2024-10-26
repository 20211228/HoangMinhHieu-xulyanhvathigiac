import cv2
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Hàm tiền xử lý ảnh và giữ ảnh ở dạng 2D
def preprocess_image(image_path, img_size=(32, 32)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    return img / 255.0  # Chuẩn hóa pixel về khoảng 0-1

# Hàm tải ảnh và gán nhãn
def load_data(dataset_path):
    X, y = [], []
    class_names = os.listdir(dataset_path)
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = preprocess_image(img_path)
            X.append(img)
            y.append(class_name)
    return np.array(X), np.array(y)

# Đọc tập dữ liệu ảnh
dataset_path = 'train/training_set/training_set'  # Thay bằng đường dẫn của bạn
X, y = load_data(dataset_path)

# Mã hóa nhãn (gán nhãn cho chuỗi thành số)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape ảnh thành dạng vector 1D
X_train_flatten = X_train.reshape(len(X_train), -1)
X_test_flatten = X_test.reshape(len(X_test), -1)

# Danh sách các mô hình
models = {
    "SVM": svm.SVC(kernel='linear'),
    "KNN": neighbors.KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": tree.DecisionTreeClassifier()
}

# Đánh giá mô hình và lưu kết quả
results = []

for name, model in models.items():
    # Đo thời gian huấn luyện
    start_time = time.time()
    model.fit(X_train_flatten, y_train)
    end_time = time.time()

    # Dự đoán và tính toán độ chính xác, precision, recall
    y_pred = model.predict(X_test_flatten)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Đếm số lượng chó và mèo được dự đoán đúng
    # correct_dogs = sum((y_pred == y_test) & (y_test == le.transform(['dog'])[0]))
    # correct_cats = sum((y_pred == y_test) & (y_test == le.transform(['cat'])[0]))

    # Lưu kết quả
    results.append({
        "Model": name,
        "Training Time (s)": end_time - start_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        # "Correct Dogs": correct_dogs,
        # "Correct Cats": correct_cats
    })

    # In báo cáo phân loại cho từng mô hình
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Hiển thị kết quả
results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)

# Hiển thị số lượng chó và mèo được phân loại chính xác cho từng mô hình
# for index, row in results_df.iterrows():
#     print(f"\nModel: {row['Model']}")
#     print(f"Correctly classified dogs: {row['Correct Dogs']}")
#     print(f"Correctly classified cats: {row['Correct Cats']}")
