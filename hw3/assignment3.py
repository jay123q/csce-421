#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
import sklearn.metrics as sk
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
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
            yPredProb = (1 + yPred) / (((1+lambda_vals[j]) * (1 - yPred)))
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
            yPredProb = (1 + yPred) / (((1+lambda_vals[j]) * (1 - yPred)))
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
    return roc_auc_score(y_test, model_R.predict(x_test))
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
    fpr, tpr, linear_threshold = roc_curve(y_test, model_L.predict(x_test))
    print("fpr, ", fpr, " tpr ", tpr)

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
        balanceFeature = None
        balanceValue = None
        balance = None
        self.left = left
        self.right = right
        # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.split_val = split_val
        self.balanceFeature = balanceFeature
        self.balanceValue = balanceValue
        self.balance = balance
        # data can be anything! we recommend dictionary with all variables you need
        self.data = data


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int, nleaves = 2) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        self.best_split = None
        self._nleaves = nleaves

    def printer(self, node: Node,  depth: int):
        " print all layers"
        if (depth == self.max_depth):
            return
        depth += 1
        print(node.split_val)
        self.printer(node.left, depth)
        self.printer(node.right, depth)
        # YOU MAY ADD ANY OTHER VARIABLES THAT YOU NEED HERE
        # YOU MAY ALSO ADD FUNCTIONS **WITHIN CLASS or functions INSIDE CLASS** TO HELP YOU ORGANIZE YOUR BETTER
        # YOUR CODE HERE

    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        if len(set(self.data[:, -1].flatten())) == 1 or self.max_depth == 0:
            return Node(split_val=np.mean(self.data[:, -1]), data=self.data[:, :-1])

        
        # Find the best split for the current data.
        best_params = self.get_best_split(self.data[:,:-1])
        
        # If no split can be found, return a leaf node with the mean value of the target variable.
        if best_params is None:
            return Node(split_val=np.mean(self.data[:, -1]), data=self.data[:,:1])
        
        # create a tree within the node and append
        left_data, right_data = self.data[best_params.left], self.data[best_params.right]
        LHS = self.__class__(left_data, max_depth=self.max_depth - 1)
        RHS = self.__class__(right_data, max_depth=self.max_depth - 1)

        left_subtree = LHS.build_tree()
        right_subtree = RHS.build_tree()
        
        # Return the current node with the best split.
        return Node(
            split_val=best_params.split_val,
            data=self.data[:,:-1],
            left=left_subtree,
            right=right_subtree,
        )
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
        ######################
        ### YOUR CODE HERE ###
        ######################
        # return 0.0
        if len(left_split) == 0 or len(right_split) == 0:
            return 1000.0
        lhsMean = np.mean(
            np.square(left_split[:, -1] - np.mean(left_split[:, -1])))
        rhsMean = np.mean(
            np.square(right_split[:, -1] - np.mean(right_split[:, -1])))
        return lhsMean+rhsMean
        pass

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively

        """


        left_data = node.data
        right_data = np.empty((0, self.data.shape[1]))

        if depth > 0 and len(set(left_data[:, -1])) != 1:
            best_split = self.get_best_split(left_data)
            if best_split:
                left_indices = best_split.left
                right_indices = best_split.right
                node.left = Node(split_val=best_split.split_val,
                                data=left_data[left_indices])
                node.right = Node(split_val=best_split.split_val,
                                data=left_data[right_indices])
                self.split(node.left, depth - 1)
                self.split(node.right, depth - 1)
            else:
                node.left = Node(split_val=np.mean(left_data[:, -1]), data=left_data)
        else:
            node.left = Node(split_val=np.mean(left_data[:, -1]), data=left_data)

    ######################
    ### YOUR CODE HERE ###
    ######################
    pass

    @typechecked
    def get_best_split(self, data: np.ndarray ) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        for i in data.shape:
          if i == 0:
            return Node(split_val=0)
        # return Node(0.0)
        # classValues = list(set(row[-1] for row in data))
        best_params = None
        best_mse = float('inf')
        
        # Iterate over all features and possible split values.
        for feature_idx in range(data.shape[1]-1):
            for split_val in np.unique(data[:, feature_idx]):
                left_data, right_data = self.one_step_split(feature_idx, split_val, data)
                
                # Skip this split if either left or right side is empty.
                if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                    continue
                
                mse = self.mean_squared_error(left_data, right_data)
                
                # Update best_params if this split has lower mse.
                if mse < best_mse:
                    best_params = (feature_idx, split_val, mse)
                    best_mse = mse
        
        if(best_params):
            left_data, right_data = self.one_step_split(best_params[0],best_params[1],data)
            node = Node(
                split_val=best_params[1],
                data=data,
                left=Node(split_val=left_data[:, -1].mean()),
                right=Node(split_val=right_data[:, -1].mean()),
            )
            return node
        return Node(split_val=np.mean(data[:,-1]), data=data)
        ######################
        ### YOUR CODE HERE ###
        ######################
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
        if len(data.shape) == 3:
          left_side = data[data[:, index, 0] < value]
          right_side = data[data[:, index, 0] >= value]
        else: 
          left_side = data[data[:, index] < value]
          right_side = data[data[:, index] >= value]
        return left_side, right_side

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
    # return False
    if node.split_val <= row[-1]:
      return False
    else:
      return True
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
    # return 0.0

    if node.left is None and node.right is None:
        return node.split_val
    if comparator(node, row):
        return predict(node.left, row, comparator)
    else:
        return predict(node.right, row, comparator)
    return 0.0
    pass


class TreeClassifier(TreeRegressor):
    def build_tree(self, depth=1) -> Node:
        # Note: You can remove this if you want to use build tree from Tree Regressor
        if depth == self.max_depth or len(self.data) < self._nleaves or len(np.unique(self.data[:, -1])) == 1:
            # Choose the most common class or the class itself as the split value
            classed, counted = np.unique(self.data[:, -1].astype(int), return_counts=True)
            return Node(split_val=int(classed[np.argmax(counted)]), data=self.data)

        # Choose the best split and return the resulting node
        return self.get_best_split(self.data)



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
        """

        ######################
        ### YOUR CODE HERE ###
        ######################
        # return 0.0
        N = left_split.shape[0] + right_split.shape[0]
        leftProb = left_split.shape[0] / N
        rightProb = right_split.shape[0] / N
        #### find the gini ####
        probLeftClass, probRightClass = [], []
        for cls in classes:
            probLeftClass.append(
                (left_split == cls).sum() / left_split.shape[0])
            probRightClass.append(
                (right_split == cls).sum() / right_split.shape[0])

        LHSClassProb = 1 - ((np.array(probLeftClass)*np.array(probLeftClass)).sum())
        RHSClassProb = 1 -((np.array(probRightClass)*np.array(probRightClass)).sum())

        return leftProb * LHSClassProb + rightProb * RHSClassProb

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """
        classValues = list(set(row[-1] for row in data))

        balanceFeature, balanceSplit, best_gini = None, None, 10000
        left_best = np.empty((1, data.shape[1]))
        right_best = np.empty((1, data.shape[1]))

        # remove 1 double compare
        for index in range(data.shape[1]-1):
            for row in np.unique(data[:, index]):
                left_best = data[data[:, index] < row]
                right_best = data[data[:, index] >= row]
                if len(left_best) < 2 or len(right_best) < 2:
                        continue
                giniCheck = self.gini_index(left_best, right_best, classValues)
                if giniCheck < best_gini:
                    best_gini = giniCheck
                    balanceFeature = index
                    balanceSplit = row

                    # balanceFeature, balanceValue, balanceScore, balanceGroup = index, row[index], mean, 

        if balanceFeature is None:
            classes, counts = np.unique(self.data[:, -1].astype(int), return_counts=True)
            return Node(split_val=int(classes[np.argmax(counts)]), data=self.data)
       # {'index':balanceFeature, 'value':balanceValue, '':balanceGroup} use these for helpers later
        left_best = data[data[:, balanceFeature] < balanceSplit]
        right_best = data[data[:, balanceFeature] >= balanceSplit]        
        
        nodeLeft = self.get_best_split(left_best)
        nodeRight = self.get_best_split(right_best)

        # print("left", left_best)
        # print("right", right_best)
        # print(data[int(balanceSplit),balanceFeature])
        node = Node(balanceSplit,
                    data, nodeLeft, nodeRight)
        return node


if __name__ == "__main__":
    # Question 1
    # filename = "Hitters.csv"  # Provide the path of the dataset
    # df = read_data(filename)
    # lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # max_iter = 1e8
    # final_features, final_label = data_preprocess(df)
    # x_train, x_test, y_train, y_test = data_split(
    #     final_features, final_label, 0.2
    # )
    # ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    # lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    # model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    # model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    # ridge_auc= ridge_area_under_curve(model_R, x_test, y_test)
    # fpr = [0., 0., 0.,0.09677419,0.09677419,0.12903226, 0.12903226 , 0.32258065 ,0.32258065, 1.]
    # tpr =  [0. , 0.04545455 , 0.86363636 , 0.86363636 , 0.90909091 , 0.90909091 ,  0.95454545 , 0.95454545 ,  1. , 1. ]
    # plt.figure()
    # plt.plot( fpr , tpr , label = "ridge prediction" )
    # plt.title(" ridge tpr vs fpr ")
    # plt.legend()

    # plt.xlabel("Fpr")
    # plt.ylabel("Tpr")
    # plt.show()
    # # Plot the ROC curve of the Ridge Model. Include axes labels,
    # # legend and title in the Plot. Any of the missing
    # # items in plot will result in loss of points.
    # ########################
    # ## Your Solution Here ##
    # ########################

    # lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)
    # fpr = [0. ,  0. , 0. ,  0.29032258 , 0.29032258 , 0.35483871 , 0.35483871 , 1. ]
    # tpr =  [0., 0.04545455, 0.90909091, 0.90909091, 0.95454545, 0.95454545, 1.,  1.  ]
    # plt.figure()
    # plt.plot( fpr , tpr ,label = "lasso prediction")
    # plt.legend()
    # plt.title(" lasso tpr vs fpr ")

    # plt.xlabel("Fpr")
    # plt.ylabel("Tpr")
    # plt.show()
    # # Plot the ROC curve of the Lasso Model.
    # # Include axes labels, legend and title in the Plot.
    # # Any of the missing items in plot will result in loss of points.
    # ########################
    # ## Your Solution Here ##
    # ########################

    # # SUB Q1
    csvname = "noisy_sin_subsample_2.csv"
    data_regress = np.loadtxt(csvname, delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
    # plt.figure()
    # plt.scatter(data_regress[:, 0], data_regress[:, 1])
    # plt.xlabel("Features, x")
    # plt.ylabel("Target values, y")
    # plt.show()

    mse_depths = []
    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        classifier = TreeClassifier(data_regress, depth)
        # print(regressor.data)
        tree = regressor.build_tree()

        mse = 0.0
        for data_point in data_regress:
            mse += (
                data_point[1]
                - predict(tree, data_point, compare_node_with_threshold)
            ) ** 2
        mse_depths.append(mse / len(data_regress))
    # regressor.printer(tree , 1)
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()

    # # SUB Q2
    # # Place the CSV file in the same directory as this notebook
    csvname = "new_circle_data.csv"
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    # plt.figure()
    # plt.scatter(
    #     data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    # )
    # plt.xlabel("Features, x1")
    # plt.ylabel("Features, x2")
    # plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        print(data_class)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(
                data_point[2]
                == predict(tree, data_point, compare_node_with_threshold)
            )
        accuracy_depths.append(correct / len(data_class))
    # # Plot the MSE
    # plt.figure()
    # plt.plot(accuracy_depths)
    # plt.xlabel("Depth")
    # plt.ylabel("Accuracy")
    # plt.show()
