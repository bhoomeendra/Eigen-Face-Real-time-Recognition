import cv2
import numpy as np


avgProjectedFace = None
eigenVectors     = None
avgFace          = None
# Loding the eigen vector , average face and average projected Face
with open('avgProjectedFace.npy', 'rb') as f:
    avgProjectedFace = np.load(f)
with open('avgFace.npy', 'rb') as f:
    avgFace = np.load(f)
with open('eigenVectors.npy', 'rb') as f:
    eigenVectors = np.load(f)

def eigenRepresentation(img, eigenVectors, dim):
    '''
    Takes a Image as a input and find its Projection in the Facespace

    img          : Input Image
    eigenVectors : List of eigen vectors sorted in Order of eigen values
    dim          : No of eigen vectors to take projection

    '''
    imgReresentation = np.zeros((dim))
    for i in range(dim):  # Loop over eigen vectors
        imgReresentation[i] = np.dot(img, eigenVectors[i]) / np.linalg.norm(img)  # Take Projection with each eigen vector
    return imgReresentation


def faceOrNot(img, eigenVectors, n_dim_pca, avgProjectedFace, avgFace, printt=False):
    '''
    Takes an image as input and producess Faceness score of the image by takes the distance from average projected face
    if the score is above 0.0001 then the image is not a face

    n_dim_pca : No. of Eigen vectore we will use to Project the image
    printt    : If we want to print the results then set it to True
    '''
    resizedImg = cv2.resize(img, (50, 37))  # Resize to match the dim of data in our Image space
    centeredImg = resizedImg.flatten() - avgFace  # Centering the Data
    projectedImg = eigenRepresentation(centeredImg, eigenVectors, n_dim_pca)  # Projecting the image to Face space
    dist = np.linalg.norm(projectedImg - avgProjectedFace)  # Calculating the Euclidean Distance
    if (printt):
        print("Distance From Average Face : ", dist)
        if (dist < 0.0001):  # The threshold we got after experimentation
            print("The Image is Face")
        else:
            print("The Image is not a Face ")
    return dist
vid = cv2.VideoCapture(0)
while (True):

    ret, frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #
    img =  gray[250:450,250:450]

    dist = faceOrNot(img,eigenVectors,100,avgProjectedFace,avgFace)
    ForNot = "Not Face"

    if(dist <0.0001):
        ForNot = "Face"
    new_image = cv2.putText(
        img=img,
        text=ForNot,
        org=(0, 50),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=2.0,
        color=(255, 255, 255),
        thickness=1
    )
    cv2.imshow('frame', new_image)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        '''
        print(img.shape)
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break
        continue
        '''
        break

vid.release()
cv2.destroyAllWindows()