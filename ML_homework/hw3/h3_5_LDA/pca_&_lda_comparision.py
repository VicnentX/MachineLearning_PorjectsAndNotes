import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# nearest-neighbor function
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# get x and y from photo
path = r"att_faces_10"
dirs = os.listdir(path)
cnt = 0
X = np.zeros((100, 10304))
index = 0

for lis in dirs:
    cnt += 1
    if lis == ".DS_Store":
        continue  # skip .DS.store file
    for img in os.listdir(path + r"/" + lis):
        pic_square = Image.open(path + r"/" + lis + r"/" + img)
        pic_square = np.array(pic_square)
        pic_line = np.reshape(pic_square, (1, -1))

        X[index] = pic_line
        index += 1

y = np.array([i // 10 for i in range(0, 100)])
print(type(y))
print(y)
print(y.size)

# get pca and lda prediction and accuracy rate
# pca d0 = 40
# lda d = 1 2 3 6 10 20 30
d0 = 40
x_d, y_lda_accuracy, y_pca_accuracy = [], [], []
for d in [1, 2, 3, 6, 10, 20, 30]:

    lda_accuracy, total_case, hit_case = 0, 0, 0

    for run in range(20):
        # stratify=y means split data according to class which is also based on test.size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # for pca
        pca = PCA(n_components=d)
        pca.fit(X_train)
        X_train_reduction = pca.transform(X_train)
        X_test_reduction = pca.transform(X_test)

        # calculate the field of classification
        for i in range(X_test_reduction.shape[0]):
            distances = dist(X_test_reduction[i], X_train_reduction)
            face_PCA_index = np.argmin(distances)
            if y_test[i] == y_train[face_PCA_index]:
                hit_case += 1
            total_case += 1

        # for lda
        pca_for_lda = PCA(n_components=40)
        pca_for_lda.fit(X_train)
        X_train_reduction_pca = pca_for_lda.transform(X_train)
        lda = LDA(n_components=d)
        lda.fit(X_train_reduction_pca, y_train)
        lda_accuracy += lda.score(pca_for_lda.transform(X_test), y_test)

    y_pca_accuracy.append(hit_case / total_case)
    y_lda_accuracy.append(lda_accuracy / 20)
    x_d.append(d)

print(y_lda_accuracy)
print(y_pca_accuracy)
print(x_d)

plt.plot(x_d, y_pca_accuracy, "*--")
plt.plot(x_d, y_lda_accuracy, "s--")
plt.xlabel('reduced dimensions d')
plt.ylabel('classification accuracy rate(%)')
plt.title('accuracy rate VS d curves')
plt.legend(['PCA', 'LDA'])
plt.show()
