# 手写识别的例子 加了噪音 看还能不能识别出来
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

digits = datasets.load_digits()
x = digits.data
y = digits.target

noisy_digits = x + np.random.normal(0, 2, size=x.shape)
example_digits = noisy_digits[y == 0, :][:10]
for num in range(1, 10):
    x_num = noisy_digits[y == num, :][:10]
    example_digits = np.vstack([example_digits, x_num])

print(example_digits.shape)


def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={"xticks":[], "yticks":[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap="binary", interpolation="nearest",
                  clim=(0, 16))
    plt.show()


plot_digits(example_digits)

# use pca reduce noise
pca = PCA(0.5)
pca.fit(noisy_digits)
print("dimens: ", pca.n_components_)


# 降噪
pca.fit(example_digits)
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)

