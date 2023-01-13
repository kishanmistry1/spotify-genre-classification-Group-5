import sys
import time

from KNN_module import KNN_Output
from RFC_module import RFC_Output
from LR_module import LR_Output
from LDA_module import LDA_Output

def run_KNN(ifname):
    print('=====Running KNN=====')
    KNN_Output(ifname)
    print('Figure has been saved as KNN Plot.jpeg')
    print('KNN Completed')
    print('')

def run_RFC(ifname):
    print('=====Running RFC=====')
    RFC_Output(ifname)
    print('Figure has been saved as RFC Plot.jpeg')
    print('RFC Completed')
    print('')

def run_LR(ifname):
    print('=====Running LR=====')
    LR_Output(ifname)
    print('Figure has been saved as LR Plot.jpeg')
    print('LR Completed')
    print('')
    
def run_LDA(ifname):
    print('=====Running LDA=====')
    LDA_Output(ifname)
    print('Figure has been saved as LDA Plot.jpeg')
    print('LDA Completed')
    print('')

def run_all(ifname):
    run_LDA(ifname)
    run_LR(ifname)
    run_KNN(ifname)
    run_RFC(ifname)
    

if __name__ == '__main__':
    """
    Flag options: 
        '-run' for running through everything
        '-RFC' for Random Forest Classifier
        '-KNN' for K-Nearest Neighbours
        '-LR' for Logisitic Regression
        '-LDA' for Fisher's Linear Discriminant
        '-h / -H / help' for help

    """

    try:
        ifname = sys.argv[1]
        flag = sys.argv[2]
    except:
        print("Sorry. Incorrect Options. First input must be file path \n2nd input options: \n\t'-run' for running through everything \n\t'-RF' for Random Forest Classifier \n\t'-LR'for Logistic Regression \n\t'-KNN' for K-Nearest Neighbours \n\t'-LinSVC' for Linear Support Vector Classifier \n\t'-h / -H / help' for help")
        sys.exit()

    if ifname[-4:] != '.csv':
        print('Sorry. Incorrect file type')
        sys.exit()
 
    elif flag == '-run':
        run_all(ifname)

    elif flag == '-RFC':
        run_RFC(ifname)
    
    elif flag == '-KNN':
        run_KNN(ifname)
    
    elif flag == '-LR':
        run_LR(ifname)
    
    elif flag == '-LDA':
        run_LDA(ifname)
        
    elif flag == '-h' or flag == '-H' or flag == 'help':
        print("First input must be file path \n2nd input options: \n\t'-run' for running through everything \n\t'-RF' for Random Forest Classifier \n\t'-LR'for Logistic Regression \n\t'-KNN' for K-Nearest Neighbours \n\t'-LinSVC' for Linear Support Vector Classifier \n\t'-h / -H / help' for help")
        
    else:
        print("Sorry. Incorrect Options. First input must be file path \n2nd input options: \n\t'-run' for running through everything \n\t'-RF' for Random Forest Classifier \n\t'-LR'for Logistic Regression \n\t'-KNN' for K-Nearest Neighbours \n\t'-LinSVC' for Linear Support Vector Classifier \n\t'-h / -H / help' for help")
        sys.exit()

