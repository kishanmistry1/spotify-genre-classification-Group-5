# Importing libraries:
import math
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#Functions imported from fomlads:
from fomlads.evaluate.eval_regression import train_and_test_partition
from fomlads.evaluate.eval_regression import train_and_test_filter
from fomlads.evaluate.eval_regression import create_cv_folds

def data_preprocessing(ifname):
    """ Function preprocesses the dataset.
    
    This involves:
    Removing uneccesary input features and any rows containing missing values.
    Grouping genres together to obtain a more balanced dataset.
    Plots displaying the number of songs for each genre are produced and saved.
    """
    
    #Uneccesary inputs and rows containing missing values are dropped.
    df = pd.read_csv(ifname)
    data = df.drop(columns=['Artist Name', 'Track Name']).dropna()
    
    #Produces plot displaying number of songs for each genre for the original dataset.
    plt.style.use('seaborn')
    data.Class.value_counts().plot(kind='bar', color='green', figsize=(8,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Number of songs for each genre. (Original Dataset)', fontsize=20)
    plt.xlabel('Genre', fontsize=16)
    plt.ylabel('Number', fontsize=16)
    
    #Plot is saved as 'Dataset (Original) Plot'
    plt.savefig('Dataset (Original) Plot.jpeg')
    plt.close()
    
    #The original dataset is modified as follows:
    #Bollywood (3) is dropped.
    #Blues (2), Alt (1), and Metal (8) are grouped.
    #Instrumental (7), Acoustic (0), Country (4) are grouped.
    #Remaining genres are left unchanged. (HipHop, Indie, Rock, Pop)

    data = data[data.Class != 3]
    data.Class.replace([0,1,2,4,5,6,7,8,9,10], [0, 1, 1, 0, 5, 6, 0, 1, 9, 10], inplace = True)
    
    #Produces plot displaying number of songs for each genre for the grouped dataset.
    plt.style.use('seaborn')
    data.Class.value_counts().plot(kind='bar', color='green', figsize=(8,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Number of songs for each genre. (Grouped Dataset)', fontsize=20)
    plt.xlabel('Genre', fontsize=16)
    plt.ylabel('Number', fontsize=16)
    
    #Plot is saved as 'Dataset (Grouped) Plot'
    plt.savefig('Dataset (Grouped) Plot.jpeg')
    plt.close()

    return data

def make_inputs_targets(data):
    """Functions separates data into inputs and targets for classification."""
    
    inputs = data.iloc[:, 0:14].to_numpy() 
    targets = data['Class'].to_numpy() 
    return inputs, targets

def make_train_val_and_test_data(inputs, targets, randomseed=0, test_fraction=0.2):
    """Function splits the data into the training+validation data and test data.
    
    Fraction of data set aside for testing is set to 0.2 by default.
    Randomseed is set to 0.
    
    """
    
    np.random.seed(randomseed)
    N = len(inputs)
    
    train_val_filter, test_filter = train_and_test_filter(N, test_fraction) 
    inputs_train_val, targets_train_val, inputs_test, targets_test = train_and_test_partition(inputs, targets, train_val_filter, test_filter)
    
    return inputs_train_val, targets_train_val, inputs_test, targets_test

def make_CV_folds(inputs_train_val, targets_train_val, randomseed=0, num_folds=5):
    """Function creates the folds for cross validation.
    
    It takes the training+validation data and creates CV folds.
    For each fold, the inputs for training and validation are scaled.
    Number of CV folds is num_folds, which is 5 by default.
    Uses funtions defined in fomlads.
    
    """
    
    np.random.seed(randomseed)
    N = len(inputs_train_val)
    folds = create_cv_folds(N, num_folds) # Creates training and validation filters for each fold.

    #Lists to store inputs (scaled) and targets for each fold.
    inputs_train_scaled_list = []
    targets_train_list = []
    inputs_val_scaled_list = []
    targets_val_list = []

    for i in range(num_folds): 
        train_filter_i = folds[i][0]
        val_filter_i = folds[i][1]
        inputs_train_i, targets_train_i, inputs_val_i, targets_val_i = train_and_test_partition(inputs_train_val, targets_train_val, train_filter_i, val_filter_i)
        
        #Inputs are scaled:
        inputs_train_scaled_i = scaler(inputs_train_i)
        inputs_val_scaled_i = scaler(inputs_val_i)
    
        inputs_train_scaled_list.append(inputs_train_scaled_i)
        targets_train_list.append(targets_train_i)
        inputs_val_scaled_list.append(inputs_val_scaled_i)
        targets_val_list.append(targets_val_i)
    
    return  inputs_train_scaled_list, targets_train_list, inputs_val_scaled_list, targets_val_list

def scaler(array):
    """Function scales the columns of the array provided.
    
    Scaled by subtracting the mean and dividing by the standard deviation.
    
    """

    array_stand = array.copy()
    for x in range(array.shape[1]):
        mean = array[:, x].mean()
        std = array[:, x].std()
        array_stand[:, x] = (array[:, x]-mean)/std
    return array_stand


