# Spotify-Genre-Classification-Group-5
INST0060 Foundations of Machine Learning - Group 5 code archive.

## Instructions
1. Open terminal.
2. Install the dependencies given in .yml file.
3. Run the program with the following command:

```bash
python3 main.py dataset.csv -flag 
```

<br>Flag options: 
<br>'-run' for running through everything 
<br>'-RFC' for Random Forest Classifier 
<br>'-KNN' for K-Nearest Neighbours
<br>'-LR' for Logisitic Regression
<br>'-LDA' for Fisher's Linear Discriminant
<br>'-h / -H / help' for help

## Structure

 `spotify-genre-classification-Group-5` is the main Python library implemented for the project.

    - `functions_module`: module containing common functions required for data preprocessing, grouping of classes, cross-validation and hyperparameter tuning.

    - `RFC_module, KNN_module, LR_module, LDA_module`: each module contains functions to tune hyperparameters (using cross-validation) and determine the F1 score on test data for optimal hyperparameter value for the respective model.

    - `main`: enables user to run all experiments using file path and flag options as inputs.
   
    - `dataset.csv`: is the original dataset. This is preprocessed and grouped to form the dataset used in experiments.
    
    - `./fomlads/`: is the supporting Python library provided in INST0060 Foundations of Machine Learning.
    
    Any figures produced in experiments are saved as .jpeg files with appropriate titles.

## Expected output

Plots showing the original dataset and grouped dataset are produced and saved. For each model, the optimal hyperparameter and final F1 score on test data are outputted. The plots showing the mean F1 score against hyperparameter values is produced and saved for each model. For Random Forest, a confusion matrix is also produced and saved.

The expected run time is about 11 minutes to run through all experiments. For individual experiments the approximate run times are:
<br> RFC : 400 seconds
<br> KNN : 180 seconds
<br> LR : 100 seconds
<br> LDA : 6 seconds

<br> Run times may vary.
