
import pandas as p
from sklearn.model_selection import train_test_split

def read_data(name):
    return p.read_csv(name)

def get_df_shape(pandaRead):
    dataFrame =  p.DataFrame(pandaRead)
    return dataFrame.shape, dataFrame

def extract_features_label(dataBlock):
    '''get the labels, and data'''
    return (dataBlock['Lag1'],dataBlock['Lag2'],dataBlock['Direction'])

def data_split(label12, direction):
    #print(train_test_split(dataFrame, shuffle=False))
    x_train, y_train, x_test, y_test = train_test_split(label12, direction, shuffle=False, stratify= None)
    return x_train,y_train,x_test,y_test


def main():
    fileName = "Smarket.csv"
    dataBlock = read_data(fileName)
    shapeTuple, dataFrame = get_df_shape(dataBlock)
    featuresLabels = extract_features_label(dataFrame)
    #x_train, y_train, x_test, y_test = data_split(featuresLabels[0] + featuresLabels[1] , featuresLabels[2] )
    x_train, y_train, x_test, y_test = data_split(featuresLabels[0] + featuresLabels[1] , featuresLabels[2] )






if ( __name__ == "__main__"):
    main()
