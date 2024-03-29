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
        img = mpimg.imread(os.path.join(folder_path , file_path))
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
    scaler = preprocessing.MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled


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
    pca = PCA(n_components=k)
    pca.fit(dataset)

    return pca, pca.explained_variance_, pca.components_


    ######################
    ### YOUR CODE HERE ###
    ######################

def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    # Calculate total sum of eigenvalues
    total_energy = np.sum(eig_values)

    # Calculate cumulative sum of eigenvalues
    cum_sum = np.cumsum(eig_values)

    # Calculate percentage of energy captured by each component
    energy_pct = cum_sum / total_energy * 100

    # Plot the curve
    plt.plot(np.arange(1, len(eig_values) + 1), eig_values, label='Eigenvalues')

    # Plot the line at 50% energy
    plt.axhline(y=total_energy / 2, color='r', linestyle='--', label='50% Energy')

    # Plot the vertical line indicating the number of components needed to capture 50% of the energy
    k = np.argmax(cum_sum >= total_energy / 2) + 1
    plt.axvline(x=k, color='g', linestyle='--', label='{} Components for 50% Energy'.format(k))

    # Set the plot title and axis labels
    plt.title('Eigenvalues vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalues')

    # Show the legend and plot
    plt.legend()
    plt.show()
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qc1_reshape_images(pca:PCA, dim_x = 243, dim_y = 320) -> np.ndarray:
    """
    reshape the pca components into the shape of original image so that it can be visualized
    """
    # Get the PCA components
    pca_components = pca.components_

    # Reshape the PCA components into the shape of the original image
    return pca_components.reshape((-1, dim_x, dim_y))
    ######################
    ### YOUR CODE HERE ###
    ######################

def qc2_plot(org_dim_eig_faces:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    fig.suptitle('10 Eigenfaces')

    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            ax[i, j].plot(org_dim_eig_faces[:, :, idx], cmap='gray')
            ax[i, j].axis('off')

    plt.show()
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.transform(dataset)
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.inverse_transform(projected_input)
    ######################
    ### YOUR CODE HERE ###
    ######################

def custom_qd3_reconstruct(dataset , org_img:np.ndarray, pca:PCA):
    """
    Simplify logic by calling other functions
    """
    # pca = PCA(n_components=comp)
    pca.fit(dataset)
    projected_input = qd1_project(org_img.reshape(1, -1),pca)
    reconstructed_input = qd2_reconstruct(projected_input,pca)
    reconstructed_input = reconstructed_input.reshape(org_img.shape)
    
    return reconstructed_input


def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    Visualize the reconstructed images using 1, 10, 20, 30, 40, 50 components
    """
    # Select a couple of images from the data
    img1 = dataset[1]
    img2 = dataset[3]
    
    components = [1, 10, 20, 30, 40, 50]
    # Reconstruct images using different numbers of components
    pca = PCA(n_components=len(components))

    reconstructed_imgs1 = [custom_qd3_reconstruct(dataset , img1, pca.set_params(n_components = k) ) for k in components]
    reconstructed_imgs2 = [custom_qd3_reconstruct(dataset, img2, pca.set_params(n_components = k) ) for k in components]


    fig, axs = plt.subplots(nrows=2, ncols=len(components), figsize=(20, 6))
    
    for i, k in enumerate(components):
        axs[0, i].imshow(reconstructed_imgs1[i].reshape(dim_x, dim_y), cmap='gray')
        axs[0, i].set_title(f'k = {k}')
        axs[1, i].imshow(reconstructed_imgs2[i].reshape(dim_x, dim_y), cmap='gray')
        axs[1, i].set_title(f'k = {k}')
    
    plt.show()

@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    # Select 5 values of k uniformly distributed in the range [1, 100]

    k_values = sorted([random.uniform(1, 101) for i in range(5)])
    best_k = 0
    highest_accuracy = 0.0

    # Loop over the k values
    for k in k_values:
        # print(k)
        # Select the first k PCA components
        pca_components = pca.components_[:int(k)]
        # Project the training data onto the first k PCA components
        projected_trainX = trainX.dot(pca_components.T)
        svm = SVC(kernel='linear')
        # Use 5-fold cross-validation to train the SVM and compute the accuracy
        kf = StratifiedKFold(n_splits=5)
        accuracies = []
        for train_idx, test_idx in kf.split(projected_trainX, trainY):
            # just like hw 3
            # break up into boxes, predict on some, score, find acc
            X_train, X_test = projected_trainX[train_idx], projected_trainX[test_idx]
            y_train, y_test = trainY[train_idx], trainY[test_idx]
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        avg_accuracy = sum(accuracies) / len(accuracies)

        # If the current accuracy is the highest, update best_k and highest_accuracy
        if avg_accuracy > highest_accuracy:
            best_k = k
            highest_accuracy = avg_accuracy

    return int(best_k), highest_accuracy
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
    best_k = None
    best_score = -1


    k_values = sorted([random.uniform(1, 101) for i in range(5)])
    scaler = StandardScaler()
    trainX_scaled = scaler.fit_transform(trainX)
    trainX_pca = pca.transform(trainX_scaled)


    for k in k_values:
        # we are paritioning k to get the parts out of it
        selected_components = trainX_pca[:, :int(k)]
        clf = Lasso()
        clf.fit(selected_components, trainY)
        score = clf.score(selected_components, trainY)
        if score > best_score:
            best_score = score
            best_k = k

    return int(best_k), best_score



if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    # qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    # qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
