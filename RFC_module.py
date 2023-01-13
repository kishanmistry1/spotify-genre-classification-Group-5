# Importing libraries:
import math
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier

from functions_module import data_preprocessing, make_inputs_targets, make_train_val_and_test_data, make_CV_folds, scaler

def RFC_hyperparameter_tuning(inputs_train_scaled_list, targets_train_list, inputs_val_scaled_list, targets_val_list, hyperparameter_list, num_folds=5):
    """Function finds the best parameter after carrying out cross validation
    and plots a figure showing the relationship between the values of the
    hyperparameter and the corresponding mean f1 score from cross validation.
    
    List of training and validation data for each fold is contained in:
    inputs_train_scaled_list, targets_train_list, inputs_val_scaled_list, targets_val_list.
    List of hyperparameters which are tested are contained in hyperparameter_list.
    Number of CV folds is num_folds, which is 5 by default.
    
    """

    mean_f1_score_list = [] #Stores the mean (across folds) f1 score for each hyperparameter value.
    ystd_errors = []
    
    for K in hyperparameter_list:
        f1_scores_K = [] #Stores the f1 scores for each fold, for a specific hyperparameter value.
        
        for i in range(num_folds): #Cross validation:
            rfc = RandomForestClassifier(n_estimators = K, class_weight = 'balanced') 
            rfc.fit(inputs_train_scaled_list[i], targets_train_list[i]) #Fit to training data.
            targets_pred_i = rfc.predict(inputs_val_scaled_list[i]) #Predict target values for validation inputs.
        
            f1_score_fold_i = f1_score(targets_val_list[i],targets_pred_i, average = 'macro')
            f1_scores_K.append(f1_score_fold_i) #f1 score for ith fold is stored.
    
        mean_f1_score_K = np.mean(f1_scores_K)#Mean f1 score (across folds).
        mean_f1_score_list.append(mean_f1_score_K) 
        ystd_errors.append(np.std(mean_f1_score_list) / np.sqrt(num_folds)) #Error bar is calculated and stored.
    
    best_parameter_f1_score = max(mean_f1_score_list) #Highest mean f1 score.
    best_parameter_index = mean_f1_score_list.index(best_parameter_f1_score)
    best_parameter = hyperparameter_list[best_parameter_index] #Best hyperparameter value.
    
    #Produces plot showing hyperparameter value vs mean f1 score on validation data.
    #Best hyperparameter value is highlighted with a red marker.
    plt.errorbar(hyperparameter_list, mean_f1_score_list,yerr = ystd_errors)
    plot(hyperparameter_list[best_parameter_index],mean_f1_score_list[best_parameter_index],'r*')
    plt.title('Number of Trees vs F1 Score on Validation Data')
    plt.xlabel('Hyperparameter - Number of Trees.')
    plt.ylabel('F1 Score on Validation Data')
    plt.savefig('RFC Plot.jpeg')
    plt.close()
    
    return best_parameter

def RFC_best_parameter(inputs_train_val, targets_train_val, inputs_test, targets_test, best_parameter):
    """Function returns the f1 score acheived (on the test data) by the model with the optimal hyperparameter.
    Produces a heatmap for the confusion matrix.
    
    Model hyperparameter is set to its optimal value (best_parameter).
    Trained using all training+validation data.
    Inputs are scaled before use.
    
    """

    inputs_train_val_scaled = scaler(inputs_train_val)
    inputs_test_scaled = scaler(inputs_test)

    rfc_best = RandomForestClassifier(n_estimators = best_parameter, class_weight = 'balanced')
    rfc_best.fit(inputs_train_val_scaled, targets_train_val)
    targets_pred = rfc_best.predict(inputs_test_scaled)
    
    #Produces heatmap of confusion matrix for test data.
    cm = confusion_matrix(targets_test,targets_pred, labels=rfc_best.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=rfc_best.classes_)
    disp.plot()
    plt.grid(False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('RFC Confusion Matrix.jpeg')
    plt.close()
    
    return f1_score(targets_test,targets_pred, average = 'macro')

def RFC_Output(ifname):
    """Function returns all outputs used in the report for RFC Model.
    
    This includes:
    The f1 score acheived (on the test data)  by the model with the optimal hyperparameter.
    A plot showing the relationship between the values of the hyperparameter
    and the corresponding mean f1 score from cross validation.
    The total run time for the RFC model.

    """

    start_time_RFC = time.time()
    
    data = data_preprocessing(ifname)
    inputs, targets = make_inputs_targets(data)

    inputs_train_val, targets_train_val, inputs_test, targets_test = make_train_val_and_test_data(inputs, targets) 
    
    inputs_train_scaled_list, targets_train_list, inputs_val_scaled_list, targets_val_list = make_CV_folds(inputs_train_val, targets_train_val)
    
    best_parameter = RFC_hyperparameter_tuning(inputs_train_scaled_list, targets_train_list, inputs_val_scaled_list, targets_val_list, hyperparameter_list=list(np.arange(50,200,10)), num_folds=5)
    
    f1_score_RFC = RFC_best_parameter(inputs_train_val, targets_train_val, inputs_test, targets_test, best_parameter)
    
    run_time_RFC = time.time() - start_time_RFC
    
    print("")
    print('Random Forest (External Model) results:')
    print('---------------------------------------')
    print('Optimal hyperparameter:', round(best_parameter,4), '(',best_parameter,')')
    print('F1-Score: ', round(f1_score_RFC,4), '(',f1_score_RFC,')')
    print('Total run time for LR: ',  round(run_time_RFC,2))

