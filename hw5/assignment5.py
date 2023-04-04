#!/usr/bin/env python
# coding: utf-8

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sklearn
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
    # Get list of image file names in the folder
    # yo I cannot understand glob glob for the life of me but I found the referencce here 
    # https://docs.python.org/3/library/glob.html
    # combine all names in the folder path wiht the ending .png
    # file_names = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    # print(file_names)
    file_names = sorted(os.listdir(folder_path))
    # Initialize arrays to store data and labels
    # x = np.empty((165, 243 * 320))
    # y = np.empty(len(file_names), dtype=int)
    x,y = [],[]
    # Read images and extract labels
    for file_path in file_names:
        # Read image and convert to grayscale
        img = mpimg.imread(file_path)
        # img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        # np.append(x,img.reshape(-1))
        # np.append(y,int(file_path[7:]))
        # x.append(img.reshape(-1))
        y.append(file_path[7:])
        x.append(img.flatten())
    # x[0] = 165
    x = np.array(x)
    
    return x, np.array(y)

@typechecked
def qa2_preprocess(dataset:np.ndarray) -> np.ndarray:
    """
    returns data (x) after pre processing

    hint: consider using preprocessing.MinMaxScaler
    """
    return dataset.dropna()   

    # df = df.drop(["NewLeague", "Player"], axis='columns')
    # nonnumericalColumn = df.select_dtypes(exclude=['int64', 'float64'])
    # wholeColumn = df.select_dtypes(include=['int64', 'float64'])
    # dummy = pd.get_dummies(nonnumericalColumn)
    # wholeFeature = pd.concat([dummy, wholeColumn], axis=1)

    # return wholeFeature, labelColumn.replace({'A': 0, 'N': 1})
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qa3_calc_eig_val_vec(dataset:np.ndarray, k:int)-> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calculate eig values and eig vectors.
    Use PCA as imported in the code to create an instance
    return them as tuple PCA, eigen_value, eigen_vector
    """



    ######################
    ### YOUR CODE HERE ###
    ######################

def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qc1_reshape_images(pca:PCA, dim_x = 243, dim_y = 320) -> np.ndarray:
    """
    reshape the pca components into the shape of original image so that it can be visualized
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

def qc2_plot(org_dim_eig_faces:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots. You can use other functions that you coded up for the assignment
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################



if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
