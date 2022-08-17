# --------------------------------------
#       PCA - EigenFaces
# --------------------------------------

import os
from PIL import Image

import numpy as np
from numpy import linalg as la

from sklearn.neighbors import KNeighborsClassifier as KN

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FaceRecognition-PCA-KNN1")

import warnings
warnings.filterwarnings("ignore")

#  Converter
# Image(bmp) to Array
def ImagetoArray(dirpath):
    images = []
    labels = []
    for label in os.listdir(dirpath):
        try:
            folder = os.path.join(dirpath,label)
            for file in os.listdir(folder):
                filepath = os.path.join(folder,file)
                with open(filepath) as f:
                    lines = f.readlines()
                    data = []
                    for line in lines[1:]:
                        if line[0] != "#":
                            data.extend([int(c) for c in line.split()])
                    images.append(np.array(data[3:]))
                    labels.append(label)
        except OSError as error:
            print(error)
    return np.matrix(images),labels


def PCA(data):
    # Matrix  Nº samples x depth --> nxd
    matrix = data.T # Convert matrix to dxn --> x

    # Calculate mean
    media = matrix.mean(axis = 1) 
    # Subtract mean from data 
    A = matrix - media # dxn

    # Calculate covariance matrix, n << d -->  200 << 10304 we apply --> 1/d*At*A size nxn
    C = 1/float(matrix.shape[0]) * A.T * A

    DeltaP, BP = la.eigh(C)
    B = A * BP
    Delta = matrix.shape[0]/float(matrix.shape[1]) * DeltaP

    # (descending order)    
    sorted_index = np.argsort(Delta)[::-1]
    B = B[:,sorted_index]

    return B/la.norm(B,axis=0),media.T


def reducePCAimage(matrix,images,media,k):
    return (images-media) * matrix[:,0:k]


if __name__ == "__main__":
    pathTest = "Test"
    pathTrain = "Train"

    # ---------------1KNN PCA----------------------
    imagesTrain,labelsTrain = ImagetoArray(pathTrain)
    imagesTest,labelsTest = ImagetoArray(pathTest)
    pca_fit_transform,media = PCA(imagesTrain)


    # -------------Code for 1KNN PCA----------------
    parameters = {
                "nbr_neighbors" : 1,
                "iteration" : 0,
            }

    tags = {
                "model" : "knn",
            }

    metrics = {

        "score" : 0,

    }
    with mlflow.start_run(nested = True):
        for i in range(1,200):
            
            imagesTrainreduced = reducePCAimage(pca_fit_transform,imagesTrain,media,i)
            imagesTestreduced = reducePCAimage(pca_fit_transform,imagesTest,media,i)

            clf = KN(n_neighbors=1)
            clf.fit(imagesTrainreduced,labelsTrain)
            result = clf.score(imagesTestreduced,labelsTest)
            
            # Register parameters and results per each iteration
            

            mlflow.log_params(parameters)

            metrics["score"] = result
            mlflow.log_metrics(metrics)
            
            mlflow.sklearn.log_model(clf, f"model")

            mlflow.set_tags(tags)




