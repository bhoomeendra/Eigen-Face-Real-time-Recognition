import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk
## Load the data



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(os.path.join(folder, filename))
        img = cv2.imread(os.path.join(folder,filename),0)

        if img is not None:
            images.append(img.flatten())
    return images

images = np.asarray(load_images_from_folder(os.path.dirname(os.path.abspath(__file__))+"\\Train_Images"))
# Calculating Average Face
avgFace =  np.mean(images,axis=0)
X_train_centered = images - avgFace
n_components = 45
pca = PCA(n_components=n_components,whiten='True',svd_solver="randomized").fit(X_train_centered)
X_train_pca = pca.transform(X_train_centered)

y_train = np.ones((X_train_centered.shape[0]))
for i in range(len(y_train)):
    y_train[i] = i//15
    print(i,y_train[i])

#Leaning The Model
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train_pca,y_train)

# Saving the learned Model
knnPickle = open('knnpickle_file', 'wb')
pk.dump(neigh, knnPickle)
#Save the learned PCA
file = open('pca_file', 'wb')
pk.dump(pca, file)

np.save('average.npy',avgFace)