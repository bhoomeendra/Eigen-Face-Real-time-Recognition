import cv2
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#Loading the Knn Model
loaded_knn = pk.load(open('knnpickle_file', 'rb'))
#Loading the PCA Model
loaded_pca = pk.load(open('pca_file', 'rb'))
#Loading the average face
avgFace = np.load('average.npy')

label2Name = {0:"Bhoomeendra" , 1:"Pavan" , 2:"Prateek"}

vid = cv2.VideoCapture(0)
while (True):

    ret, frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    img =  gray[250:450,250:450]
    #Performing the mean centering and then PCA
    pcaImg = loaded_pca.transform([img.flatten() - avgFace])
    #Performing the
    label = loaded_knn.predict(pcaImg)



    ForNot = "Face"
    new_image = cv2.putText(
        img=img,
        text=label2Name[int(label)],
        org=(0, 50),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.7,
        color=(255, 255, 255),
        thickness=1
    )
    cv2.imshow('frame', new_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()