import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_2D_dataset():
    data = scipy.io.loadmat('Dataset/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    
    return train_X, train_Y, test_X, test_Y

def load_2D_dataset_pandas(train_X, train_Y, test_X, test_Y):
    """Retrieve 2 features data (x1,x2) and 1 label (y) of each type of dataset
    -Merge to training dataset and test set
    -Categorize y into category
    -Plot Training & Test set
    """
    df_train = pd.DataFrame(np.column_stack([train_X.T, train_Y.T]), 
                               columns=['x1', 'x2', 'y'])

    df_test = pd.DataFrame(np.column_stack([test_X.T, test_Y.T]), 
                               columns=['x1', 'x2', 'y'])

    df_train = df_train.astype({"y":'category'})
    df_test = df_test.astype({"y":'category'})
    
    
    sns.set()
    fig, axes = plt.subplots(1, 2)

    p = sns.scatterplot(data=df_train,x='x1',y='x2',hue='y',ax=axes[0])
    p.set_title('Training set')

    g = sns.scatterplot(data=df_test,x='x1',y='x2',hue='y',ax=axes[1])
    g.set_title('Test set')

    fig.tight_layout()
    
    return df_train, df_test
