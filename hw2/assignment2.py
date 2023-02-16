################
################
# Q1
################
################
import assignment2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
import scipy.stats


# Download and read the data.
def read_data(filename: str) -> pd.DataFrame:
    '''
        read data and return dataframe
    '''
    ########################
    ## Your Solution Here ##
    ########################
    return pd.read_csv(filename)
    pass


# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    dataTrain = df_train['x']
    labelTrain = df_train['y']
    dataTest = df_test['x']
    labelTest = df_test['y']

   # print("train data ",dataTrain," label 1 test ",labelTest," data test ",dataTest," label test 2 ",labelTest)
    return dataTrain, labelTrain, dataTest, labelTest
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Implement LinearRegression_Local class


class LinearRegression_Local:
    def __init__(self, learning_rate=0.00001, iterations=30):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # weight initialization

        self.m, self.n = X.shape
        #print(X.shape)
        self.W = np.zeros(self.n)
        # data
        self.X = X
        self.Y = Y
        self.b = 0

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()

        # print(self.W.shape)
        # print(self.Y.shape)
        # print(self.b.shape)
        return self
    
    # Helper function to update weights in gradient descent
    def update_weights(self):
        
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        # print("iterations ", i)

        # print("dw shape ", dW.shape)
        # print(" w shape ",self.W.shape)
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        # predict on data and calculate gradients
        #print("X shape ",X.shape)
        return X.dot(self.W) + self.b
        # YOUR CODE HERE
        # YOUR CODE HERE

    ########################
    ## Your Solution Here ##
    ########################
    pass

# Build your model


def build_model(train_X: np.array, train_y: np.array):
    '''
        Instantiate an object of LinearRegression_Local class, train the model object
        using training data and return the model object
    '''


    # print(train_X.shape, train_y.shape)
    linearModel = LinearRegression_Local()
    # train_X = np.expand_dims(train_X, -1)
    linearModel.fit(train_X, train_y)

    return linearModel
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Make predictions with test set


def pred_func(model, X_test):
    '''
        return numpy array comprising of prediction on test set using the model
    '''
    # X_test = np.expand_dims(X_test, axis=1)
    # print(X_test.shape)
    # print(X_test.shape)
    return model.predict(X_test)
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Calculate and print the mean square error of your prediction


def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    #y_test = np.expand_dims(y_test, -1)

    return metrics.mean_squared_error(y_test, pred)
    ########################
    ## Your Solution Here ##
    ########################
    pass

################
################
# Q2
################
################

# Download and read the data.


def read_training_data(filename: str) -> tuple:
    '''
        read train data into a dataframe df1, store the top 10 entries of the dataframe in df2
        and return a tuple of the form (df1, df2, shape of df1)   
    '''
    df1 = pd.read_csv(filename)
    #print(df1)
    df2 = df1[0:10]

    return (df1, df2, df1.shape)
    ########################
    ## Your Solution Here ##
    ########################
    pass

# Prepare your input data and labels


def data_clean(df_train: pd.DataFrame) -> tuple:
    '''
        check for any missing values in the data and store the missing values in series s, drop the entries corresponding 
        to the missing values and store dataframe in df_train and return a tuple in the form: (s, df_train)
    '''
    s = df_train.isnull().sum()
    df_train = df_train.dropna()
    return s, df_train
    ########################
    ## Your Solution Here ##
    ########################
    pass


def feature_extract(df_train: pd.DataFrame) -> tuple:
    '''
        New League is the label column.
        Separate the data from labels.
        return a tuple of the form: (features(dtype: pandas.core.frame.DataFrame), label(dtype: pandas.core.series.Series))
    '''
    labelColumn = df_train['NewLeague']
    df_train = df_train.drop(columns="NewLeague")
    return df_train, labelColumn
    ########################
    ## Your Solution Here ##
    ########################
    pass


def data_preprocess(feature: pd.DataFrame) -> pd.DataFrame:
    '''
        Separate numerical columns from nonnumerical columns. (In pandas, check out .select dtypes(exclude = ['int64', 'float64']) and .select dtypes(
        include = ['int64', 'float64']). Afterwards, use get dummies for transforming to categorical. Then concat both parts (pd.concat()).
        and return the concatenated dataframe.
    '''
    nonnumericalColumn = feature.select_dtypes(exclude=['int64', 'float64'])
    wholeColumn = feature.select_dtypes(include=['int64', 'float64'])
    dummy = pd.get_dummies(nonnumericalColumn)
    wholeFeature = pd.concat([dummy, wholeColumn], axis = 1)
    return wholeFeature
    ########################
    ## Your Solution Here ##
    ########################
    pass


def label_transform(labels: pd.Series) -> pd.Series:
    '''
        Transform the labels into numerical format and return the labels
    '''
    return labels.replace({'A': 0, 'N': 1})
    ########################
    ## Your Solution Here ##
    ########################
    pass

################
################
# Q3
################
################


def data_split(features: pd.DataFrame, label: pd.Series, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set using the given random state
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    return train_test_split(features, label, test_size=0.2, random_state=random_state)

    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray):
    '''
        Instantiate an object of LinearRegression_Local class, train the model object
        using training data and return the model object
    '''
    # x_train = np.expand_dims(x_train, -1)
    # y_train = np.expand_dims(y_train, -1)
    # print(x_train.shape)
    # print(y_train.shape)

    linearModel = LinearRegression()
    linearModel.fit(x_train, y_train)
    return linearModel

    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, max_iter=1000000):
    '''
        Instantiate an object of LogisticRegression class, train the model object
        use provided max_iterations for training logistic model
        using training data and return the model object
    '''
    # x_train = np.expand_dims(x_train, -1)
    # y_train = np.expand_dims(y_train, -1)
    logisticalRegression = LogisticRegression(max_iter=max_iter)
    logisticalRegression.fit(x_train, y_train,)
    return logisticalRegression
    ########################
    ## Your Solution Here ##
    ########################
    pass


def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return the tuple consisting the coefficients for each feature for Linear Regression 
        and Logistic Regression Models respectively
    '''
    linearCoe = np.array(linear_model.coef_)
    logisticalCoe = np.array(logistic_model.coef_)
    return linearCoe, logisticalCoe
    ########################
    ## Your Solution Here ##
    ########################
    pass


def linear_pred_and_area_under_curve(linear_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    # linear_model.fit(x_test,y_test)
    y_pred = linear_model.predict(x_test)

    fpr,tpr,linear_threshold = metrics.roc_curve( y_test, y_pred )
    # predict = precision_score( y_test , y_pred_prob )
    reg_area = metrics.roc_auc_score( y_test , y_pred )
    #print(predict,fpr,tpr,linear_threshold,reg_area)
    return y_pred,fpr,tpr,linear_threshold,reg_area
    ########################
    ## Your Solution Here ##
    ########################
    pass


def logistic_pred_and_area_under_curve(logistic_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [logReg_pred, logReg_fpr, logReg_tpr, log_threshold, logReg_area_under_curve]
        Finally plot the ROC Curve
    '''
    # logistic_model.fit(x_test,y_test)
    y_pred_prob = logistic_model.predict_proba(x_test)[:,1]

    fpr,tpr,logisticThreshold = metrics.roc_curve( y_test, y_pred_prob)
    # predict = precision_score( y_test , y_pred_prob )
    reg_area = metrics.roc_auc_score( y_test , y_pred_prob)
    #print(predict,fpr,tpr,linear_threshold,reg_area)
    return y_pred_prob,fpr,tpr,logisticThreshold,reg_area
    ########################
    ## Your Solution Here ##
    ########################
    pass


def optimal_thresholds(linear_threshold: np.ndarray, linear_reg_fpr: np.ndarray, linear_reg_tpr: np.ndarray, log_threshold: np.ndarray, logReg_fpr: np.ndarray, logReg_tpr: np.ndarray) -> Tuple[float, float]:
    '''
        return the tuple consisting the thresholds of Linear Regression and Logistic Regression Models respectively
    '''
    linearMax = np.argmax(
        linear_reg_tpr - linear_reg_fpr)
    logMax = np.argmax(
        logReg_tpr - logReg_fpr)
    return linear_threshold[linearMax], log_threshold[logMax]
    ########################
    ## Your Solution Here ##
    ########################
    pass


def stratified_k_fold_cross_validation(num_of_folds: int, shuffle: True, features: pd.DataFrame, label: pd.Series):
    '''
        split the data into 5 groups. Checkout StratifiedKFold in scikit-learn
    '''
    return StratifiedKFold(n_splits=num_of_folds, shuffle=shuffle, random_state=None )
    ########################
    ## Your Solution Here ##
    ########################
    pass


def train_test_folds(skf, num_of_folds: int, features: pd.DataFrame, label: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    '''
        train and test in for loop with different training and test sets obatined from skf. 
        use a PENALTY of 12 for logitic regression model for training
        find features in each fold and store them in features_count array.
        populate aucLog and aucLinear arrays with roc_auc_score of each set trained on logistic regression and linear regression models respectively.
        populate f1_dict['log_reg'] and f1_dict['linear_reg'] arrays with f1_score of trained logistic and linear regression models on each set
        return features_count, aucLog, aucLinear, f1_dict dictionary
    '''
    features_count = np.array([])
    aucLog = np.array([])
    aucLinear = np.array([])
    f1_dict = {'log_reg':[], 'linear_reg':[]}


    linearReg = LinearRegression()
    logReg = LogisticRegression(penalty='l2', max_iter=1000000)

    for trainIndex, testIndex in skf.split(features, label):
        featuresTrain, featuresTest = features.iloc[trainIndex], features.iloc[testIndex]
        # asked ta
        labelTrain, labelTest = label.iloc[trainIndex].values.ravel(), label.iloc[testIndex].values.ravel()
        # get the y of training 
        features_count = np.append(features_count, featuresTrain.shape[1])

        logReg.fit(featuresTrain, labelTrain)

        aucLog = np.append(aucLog, metrics.roc_auc_score(labelTest, logReg.predict(featuresTest).clip(0).astype(int)))

        f1_dict['log_reg'].append(
            metrics.f1_score(labelTest, logReg.predict(featuresTest).clip(0).astype(int)
                             ))
 
        linearReg.fit(featuresTrain, labelTrain)

        aucLinear = np.append(aucLinear, metrics.roc_auc_score(labelTest, linearReg.predict(featuresTest).round().clip(0).astype(int)
))

        f1_dict['linear_reg'].append(metrics.f1_score(labelTest, linearReg.predict(featuresTest).round().clip(0).astype(int)
))


    return features_count, aucLog, aucLinear, f1_dict
    pass


def is_features_count_changed(features_count: np.array) -> bool:
    '''
        compare number of features in each fold (features_count array's each element)
        return true if features count doesn't change in each fold. else return false
    '''
    linearReg = LinearRegression()
    logReg = LogisticRegression(penalty='l2', max_iter=1000000)



    return True
    ########################
    ## Your Solution Here ##
    ########################
    pass


def mean_confidence_interval(data: np.array, confidence=0.95) -> Tuple[float, float, float]:
    '''
        To calculate mean and confidence interval, in scipy checkout .sem to find standard error of the mean of given data (AUROCs/ f1 scores of each model, linear and logistic trained on all sets). 
        Then compute Percent Point Function available in scipy and mutiply it with standard error calculated earlier to calculate h. 
        The required interval is from mean-h to mean+h
        return the tuple consisting of mean, mean -h, mean+h
    '''
    print(data)
    mean = np.mean( data )
    print(mean)
    standardError = scipy.stats.sem( data, axis = 0 ,  ddof = 0  )

    h = scipy.stats.norm.ppf( confidence , mean ) * standardError
    return mean, mean-h, mean+h

    ########################
    ## Your Solution Here ##
    ########################
    pass


if __name__ == "__main__":

    ###############
    ################
    # Q1
    ################
    ################
    data_path_train = "LinearRegression/train.csv"
    data_path_test = "LinearRegression/test.csv"
    df_train, df_test = read_data(data_path_train), read_data(data_path_test)
    # print(df_train.head(n=10))
    # print(df_test.head(n=10))
    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    # print(train_X.shape)
    # print(train_y.shape)

    model = build_model(train_X, train_y)
    preds = pred_func(model, test_X)
    # Make prediction with test set

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)

    #plot your prediction and labels, you can save the plot and add in the report

    # plt.plot(test_y, label='label')
    # plt.plot(preds, label='pred')
    # plt.legend()
    # plt.show()

    ################
    ################
    # Q2
    ################
    ################

    data_path_training = "Hitters.csv"

    train_df, df2, df_train_shape = read_training_data(data_path_training)
    s, df_train_mod = data_clean(train_df)
    features, label = feature_extract(df_train_mod)
    final_features = data_preprocess(features)
    final_label = label_transform(label)
    # print(final_label , "Asdd ")

    ################
    ################
    # Q3
    ################
    ################

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    aucLog = []
    aucLinear = []
    features_count = []
    f1_dict = {'logReg': [], 'linear_reg': []}
    # is_features_count_changed = True

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(
        linear_model, logistic_model)

    # print("Coe ", linear_coef, logistic_coef )

    linear_y_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve = linear_pred_and_area_under_curve(
        linear_model, X_test, y_test)

    log_y_pred, logReg_fpr, logReg_tpr, log_threshold, logReg_area_under_curve = logistic_pred_and_area_under_curve(
        logistic_model, X_test, y_test)

    # plt.plot(logReg_fpr, logReg_tpr, label='logistic')
    # plt.plot(linear_reg_fpr, linear_reg_tpr, label='linear')
    # plt.legend()
    # plt.show()

    linear_optimal_threshold, log_optimal_threshold = optimal_thresholds(
        linear_threshold, linear_reg_fpr, linear_reg_tpr, log_threshold, logReg_fpr, logReg_tpr)
    print("Optimal threshold  ", linear_optimal_threshold , log_optimal_threshold )
    skf = stratified_k_fold_cross_validation(
        num_of_folds, True,  final_features, final_label)
    features_count, aucLog, aucLinear, f1_dict = train_test_folds(
        skf, num_of_folds, final_features, final_label)

    print("Does features change in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    is_features_count_changed_box = is_features_count_changed(features_count)

    print(is_features_count_changed_box)

    aucLinear_mean, aucLinear_open_interval, aucLinear_close_interval = 0, 0, 0
    aucLog_mean, aucLog_open_interval, aucLog_close_interval = 0, 0, 0

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = 0, 0, 0
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = 0, 0, 0

    # Find mean and 95% confidence interval for the AUROCs for each model and populate the above variables accordingly
    # Hint: use mean_confidence_interval function and pass roc_auc_scores of each fold for both models (ex: aucLog)
    # Find mean and 95% confidence interval for the f1 score for each model.

    aucLinear_mean, aucLinear_open_interval, aucLinear_close_interval = assignment2.mean_confidence_interval(
        aucLinear)
    print("auc score linear ",  aucLinear_mean, aucLinear_open_interval, aucLinear_close_interval )


    aucLog_mean, aucLog_open_interval, aucLog_close_interval = assignment2.mean_confidence_interval(
        aucLog)
    print("auc score log ",  aucLinear_mean, aucLinear_open_interval, aucLinear_close_interval )


    f1_linear_mean, f1_linear_open_interval, f1_linear_close_intervel = assignment2.mean_confidence_interval(
        f1_dict['linear_reg'])    
    print("auc score linear f1 ",  f1_linear_mean, f1_linear_open_interval, f1_linear_close_intervel)


    f1_log_mean, f1_log_open_interval, f1_log_close_interval = assignment2.mean_confidence_interval(
        f1_dict['log_reg'])
    print("auc score log f1 ", f1_log_mean, f1_log_open_interval, f1_log_close_interval )
