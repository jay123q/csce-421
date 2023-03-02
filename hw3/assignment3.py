#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score, mean_squared_error
import random
from typeguard import typechecked

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    return pd.read_csv(filename)
    ########################
    ## Your Solution Here ##
    ########################
    pass


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submiited for problem 2 of HW2
    """
    df = df.dropna()

    # df = df.dropna()
    labelColumn = df.loc[:, 'NewLeague']

    df = df.drop(["NewLeague", "Player"], axis='columns')
    nonnumericalColumn = df.select_dtypes(exclude=['int64', 'float64'])
    wholeColumn = df.select_dtypes(include=['int64', 'float64'])
    dummy = pd.get_dummies(nonnumericalColumn)
    wholeFeature = pd.concat([dummy, wholeColumn], axis=1)

    return wholeFeature, labelColumn.replace({'A': 0, 'N': 1})
    #######################
    ## Your Solution Here ##
    ########################
    pass


@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    return tuple(train_test_split(features, label, test_size=test_size))

    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_predict(object, x_train, y_train, x_test):

    object.fit(x_train, y_train)
    yPred = object.predict(x_test)
    return yPred


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary
    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"Ridge": {1e-3: [], 1e-2: [], 1e-1: [],
                      1: [], 1e1: [], 1e2: [], 1e3: []}}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n):
        for j in range(len(lambda_vals)):
            ridgeReg = Ridge(alpha=lambda_vals[j], max_iter=max_iter)
            yPred = train_predict(ridgeReg, x_train, y_train, x_test)
            yPredProb =  (1 + yPred) / (((1+lambda_vals[j]) * (1 - yPred)))
            aucs["Ridge"][lambda_vals[j]].append(
                roc_auc_score(y_test, yPredProb))

    print("Ridge mean AUCs:")
    ridge_mean_auc = {}
    lasso_aucs = pd.DataFrame(
        aucs["Ridge"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        ridge_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    return ridge_mean_auc


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"lasso": {1e-3: [], 1e-2: [], 1e-1: [],
                      1: [], 1e1: [], 1e2: [], 1e3: []}}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    ########################
    ## Your Solution Here ##
    ########################
    for i in range(n):
        for j in range(len(lambda_vals)):
            lassReg = Lasso(alpha=lambda_vals[j], max_iter=max_iter)
            yPred = train_predict(lassReg, x_train, y_train, x_test)
            yPredProb = (1 + yPred)/ (((1+lambda_vals[j]) * (1 - yPred)))
            aucs["lasso"][lambda_vals[j]].append(
                roc_auc_score(y_test, yPredProb))

    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs = pd.DataFrame(aucs["lasso"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    return lasso_mean_auc


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ridgeReg = Ridge(alpha=optimal_alpha, max_iter=max_iter)
    ridgeReg.fit(x_train, y_train)
    return ridgeReg, ridgeReg.coef_
    ########################
    ## Your Solution Here ##
    ########################
    pass


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    ########################
    ## Your Solution Here ##
    ########################
    lassoReg = Lasso(alpha=optimal_alpha, max_iter=max_iter)
    lassoReg.fit(x_train, y_train)
    return lassoReg, lassoReg.coef_
    pass


@typechecked
def ridge_area_under_curve(
    model_R, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    # fpr,tpr,linear_threshold = roc_curve( y_test, model_R.predict(x_test) )
    # print("fpr, ", fpr , " tpr ", tpr  )    
    return roc_auc_score(y_test, model_R.predict(x_test) )
    ########################
    ## Your Solution Here ##
    ########################
    pass


@typechecked
def lasso_area_under_curve(
    model_L, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_aplha

    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    fpr,tpr,linear_threshold = roc_curve( y_test, model_L.predict(x_test) )
    print("fpr, ", fpr , " tpr ", tpr  )

    return roc_auc_score(y_test, model_L.predict(x_test))
    ########################
    ## Your Solution Here ##
    ########################
    pass


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)
        balanceIndex = None
        balanceValue = None
        balanceGroups = None
        self.left = left
        self.right = right
        # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.split_val = split_val
        self.balanceIndex = balanceIndex
        self.balanceValue = balanceValue
        self.balanceGroups = balanceGroups
        # data can be anything! we recommend dictionary with all variables you need
        self.data = {'left': left, 'right': right, 'index': balanceIndex,
                     'value': balanceValue, 'groups': balanceGroups}


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        self.root = None
        self.best_split = None
        self._nleaves = None
        print(data)
    # def AddNode(self, pointer = self.root, node: Node ):
    #     # using the stump formula
    #     if (node.data <= pointer.data):
    #         # check left
    #         if (self.left == None):
    #             # if its none add
    #             self.left = pointer
    #         else:
    #             self = self.left
    #             TreeRegressor.AddNode(pointer, node)
    #     else:
    #         if (self.right == None):
    #             self.right = pointer
    #         else:
    #              self = self.right
    #              TreeRegressor.AddNode(node)

        # YOU MAY ADD ANY OTHER VARIABLES THAT YOU NEED HERE
        # YOU MAY ALSO ADD FUNCTIONS **WITHIN CLASS or functions INSIDE CLASS** TO HELP YOU ORGANIZE YOUR BETTER
        # YOUR CODE HERE
        self.bestSplit = self.get_best_split(data)

    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """

        min_size = 0
        self.split(self.root, self.max_depth, min_size, 1)
        return self.root
        # root.split_value = TreeRegressor.mean_squared_error()
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    @typechecked
    def mean_squared_error(
        self, left_split: np.ndarray, right_split: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split 
        """
        print(left_split.shape)
        print(right_split.shape)
        return mean_squared_error(left_split, right_split)
        mse = np.mean(np.square(left_split - np.mean(left_split)))

        mse += np.mean(np.square(right_split - np.mean(right_split)))
        ######################
        ### YOUR CODE HERE ###
        ######################
        return np.mean(np.square(left_split - right_split)) + np.mean(np.square(right_split - left_split))
        pass

    @typechecked
    def split(self, node: Node, depth: int, counter: int) -> None:
        """
        Do the split operation recursively

        """
        print("aaaa")
        # create a value for a decision tree

        # this is going to count down
        # counter = 1
        # if (counter == 1):
        #         self.root = TreeRegressor.get_best_split(node.data.sort())
        #         head = self.root
        #         counter += 1
        #         headLeft = head.left
        #         headRight = head.right
        #         TreeRegressor.split( headLeft , depth , counter )
        #         TreeRegressor.split( headRight , depth , counter  )
        # elif( counter == depth ):
        #     return
        # else:
        #         counter += 1
        #         nodeLeft = node.left
        #         nodeRight = node.right
        #         headLeft = TreeRegressor.get_best_split(nodeLeft.data.sort())
        #         headRight = TreeRegressor.get_best_split(nodeRight.data.sort())
        #         TreeRegressor.split( headLeft , depth , counter  )
        #         TreeRegressor.split( headRight , depth , counter )

        # if( self.root == None ):
        #     self.root = node
        #     self.root.data = self.root.data.sort()
        #     self.root.split_val = self.root.data[(len(self.root.data))/2]

        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        features, label = data[:,0],data[:,1]
        nRow, nCol = data.shape
        checkVal = -1
        for dataPoint in range(nCol):
            xCurr = data[:, dataPoint]
            for threshold in np.unique(xCurr):
                #dataFrame = np.concatenate((features, label.reshape(1,-1).T) , axis=0 )
                dataFrame = data
                print(dataFrame.shape)
                dataFrame = data
                leftHalf  = np.array([row for row in dataFrame if row[dataPoint]  <= threshold ])
                rightHalf  = np.array([row for row in dataFrame if row[dataPoint]  > threshold ])
                if len( leftHalf ) > 0 and len( rightHalf ) > 0:
                    data = dataFrame[:,-1]
                    yLeft = leftHalf[:,-1]
                    yRight = rightHalf[:,-1]
                
                    yLeft = np.expand_dims(yLeft,axis = 1)
                    yRightShape = yRight.shape
                    #yLeft = yLeft.reshape(yRight.shape)
                    print(yLeft.shape," left and right ", yRight.shape )
                    mean = self.mean_squared_error( yLeft , yRight )

                    if( mean > checkVal ):
                        node = Node( split_val=threshold, data= data )
                        checkVal = mean
            return node
                    
                # classValues = list(set(row[-1] for row in data))
                # balanceIndex, balanceValue, balanceScore, balanceGroup = 999, 999, 999, None
                # #
                # for index in range(len(data)-1):
                #     for row in data:
                #         groups = self.one_step_split(index, row[index], data)
                #         print(groups)
                #         mean = self.mean_squared_error( groups )
                #         if mean < balanceScore :
                #             balanceIndex, balanceValue, balanceScore, balanceGroup = index, row[index], mean, groups

                # return {'index':balanceIndex, 'value':balanceValue, 'groups':balanceGroup}
                ######################
                ### YOUR CODE HERE ###
                ######################
        return Node(split_val=0)
        pass

    @typechecked
    def one_step_split(
        self, index: int, value: float, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        left, right = [], []
        for row in data:
            # print(row[index]," compared too ", value)
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return np.array(left), np.array(right)

        ######################
        ### YOUR CODE HERE ###
        ######################
        pass


@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
    return (node.split_val > row[len(row)])
    ######################
    ### YOUR CODE HERE ###
    ######################
    pass


@typechecked
def predict(
    node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]
) -> float:
    ######################
    ### YOUR CODE HERE ###
    ######################

    pass


class TreeClassifier(TreeRegressor):
    def build_tree(self):
        # Note: You can remove this if you want to use build tree from Tree Regressor
        ######################
        ### YOUR CODE HERE ###
        ######################
        min_size = 0
        self.split(self.root, self.max_depth, min_size, 1)
        return self.root
        pass

    @typechecked
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead
        # """
        # # count all samples at split point
        # n_instances = float(sum([len(group) for group in left_split]))
        # n_instances += float(sum([len(group) for group in right_split]))
        # # sum weighted Gini index for each group
        # gini = 0.0
        # # make one group
        # groups = left_split + right_split
        # for group in groups:
        #     size = float(len(group))
        #     # avoid divide by zero
        #     if size == 0:
        #         continue
        #     score = 0.0
        #     # score the group based on the score for each class
        #     for class_val in classes:
        #         p = [row[-1] for row in group].count(class_val) / size
        #         score += p * p
        #     # weight the group score by its relative size
        #     gini += (1.0 - score) * (size / n_instances)
        # return gini
        pass

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """

        # classValues = list(set(row[-1] for row in data))
        # balanceIndex, balanceValue, balanceScore, balanceGroup = 999, 999, 999, None
        # #
        # for index in range(len(data)-1):
        #     for row in data:
        #         groups = self.one_step_split(index, row[index], data)
        #         print(groups)
        #         mean = self.gini_index(groups[0], groups[1],  classValues)
        #         if mean < balanceScore:
        #             balanceIndex, balanceValue, balanceScore, balanceGroup = index, row[
        #                 index], mean, groups

        return balanceValue
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass


if __name__ == "__main__":
    # Question 1
    filename = "Hitters.csv"  # Provide the path of the dataset
    df = read_data(filename)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(
        final_features, final_label, 0.2
    )
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc= ridge_area_under_curve(model_R, x_test, y_test)
    fpr = [0., 0., 0.,0.09677419,0.09677419,0.12903226, 0.12903226 , 0.32258065 ,0.32258065, 1.]  
    tpr =  [0. , 0.04545455 , 0.86363636 , 0.86363636 , 0.90909091 , 0.90909091 ,  0.95454545 , 0.95454545 ,  1. , 1. ]
    plt.figure()
    plt.plot( fpr , tpr )
    plt.title(" ridge tpr vs fpr ")
    plt.xlabel("Fpr")
    plt.ylabel("Tpr")
    plt.show()
    # Plot the ROC curve of the Ridge Model. Include axes labels,
    # legend and title in the Plot. Any of the missing
    # items in plot will result in loss of points.
    ########################
    ## Your Solution Here ##
    ########################

    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    fpr = [0. ,  0. , 0. ,  0.29032258 , 0.29032258 , 0.35483871 , 0.35483871 , 1. ]
    tpr =  [0., 0.04545455, 0.90909091, 0.90909091, 0.95454545, 0.95454545, 1.,  1.  ]
    plt.figure()
    plt.plot( fpr , tpr )
    plt.title(" lasso tpr vs fpr ")
    plt.xlabel("Fpr")
    plt.ylabel("Tpr")
    plt.show()
    # Plot the ROC curve of the Lasso Model.
    # Include axes labels, legend and title in the Plot.
    # Any of the missing items in plot will result in loss of points.
    ########################
    ## Your Solution Here ##
    ########################

    # # SUB Q1
    csvname = "noisy_sin_subsample_2.csv"
    data_regress = np.loadtxt(csvname, delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()

    # mse_depths = []
    # for depth in range(1, 5):
    #     regressor = TreeRegressor(data_regress, depth)
    #     classifier = TreeClassifier(data_regress, depth)
    #     tree = regressor.build_tree()
    #     mse = 0.0
    #     for data_point in data_regress:
    #         mse += (
    #             data_point[1]
    #             - predict(tree, data_point, compare_node_with_threshold)
    #         ) ** 2
    #     mse_depths.append(mse / len(data_regress))
    # plt.figure()
    # plt.plot(mse_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("MSE")
    # plt.show()

    # # SUB Q2
    # # Place the CSV file in the same directory as this notebook
    # csvname = "new_circle_data.csv"
    # data_class = np.loadtxt(csvname, delimiter=",")
    # data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    # plt.figure()
    # plt.scatter(
    #     data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    # )
    # plt.xlabel("Features, x1")
    # plt.ylabel("Features, x2")
    # plt.show()

    # accuracy_depths = []
    # for depth in range(1, 8):
    #     print(data_class)
    #     tree = classifier.build_tree()
    #     correct = 0.0
    #     for data_point in data_class:
    #         correct += float(
    #             data_point[2]
    #             == predict(tree, data_point, compare_node_with_threshold)
    #         )
    #     accuracy_depths.append(correct / len(data_class))
    # # Plot the MSE
    # plt.figure()
    # plt.plot(accuracy_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("Accuracy")
    # plt.show()
