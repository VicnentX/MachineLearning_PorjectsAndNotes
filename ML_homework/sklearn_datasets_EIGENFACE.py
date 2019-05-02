import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people()
print(faces.keys())
print(faces.data.shape)
print(faces.images.shape)

random_indices = np.random.permutation(len(faces.data))
x = faces.data[random_indices]

example_faces = x[:36, :]
print("example_faces.shape: ", example_faces.shape)


def plot_faces(faces):

    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47),
                  cmap="bone")
    plt.show()


plot_faces(example_faces)
print(faces.target[random_indices])

# EIGENFACE

pca = PCA(svd_solver="randomized")
pca.fit(x)

# 画出特征脸(越往后细节越多)
plot_faces(pca.components_)

# 比如有些人在这个数据库照片太少了 只取出大于60张照片的人作为样本
faces2 = fetch_lfw_people(min_faces_per_person=60)
print(faces2.data.shape)
