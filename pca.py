import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cv2

iris = datasets.load_iris()
data = iris.data
target = iris.target


def standardization(_data):
    return _data - _data.mean(axis=0)


def covariance(_metric):
    return _metric.T.dot(_metric)


def PCA_de(_data, maxCompoents=None):
    lg = np.linalg.eigh(covariance(standardization(_data)))
    if maxCompoents:
        col = lg[1].shape[0]
        eigenvectors = np.fliplr(lg[1][:, col - maxCompoents:])
        eigenvalue = lg[0][::-1][:maxCompoents]
    else:
        eigenvectors = np.fliplr(lg[1])
        eigenvalue = lg[0][::-1]
    return eigenvectors, eigenvalue


eigvector, eigvalue = PCA_de(data, maxCompoents=2)
mu, eig = cv2.PCACompute(data, np.array([]), maxComponents=2)

print(eig.T, eigvector)
data_y = data.dot(eig.T)
print(data_y.shape)

plt.figure()
plt.scatter(data_y[:, 0], data_y[:, 1], c=target, cmap=plt.cm.Paired, s=10)
plt.axis([2, 10, 4, 7])
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.show()












