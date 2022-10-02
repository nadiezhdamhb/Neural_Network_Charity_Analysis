# Neural_Network_Charity_Analysis

## Analysis Overview:

### Background & Purpose

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Data Preprocessing
  1.  The columns labeled "EIN" and "NAME" were dropped from the dataset as they are identification information and will not be used as features.  
  2.  The column "IS_SUCCESSFUL" contains binary data that helps to classify if the company effectively used the donation received from Alphabet Soup.  This column will be used as the target and dropped from the dataset as well.  
  3.  The remaining columns will be used for the features of the machine learning model.  "APPLICATION_TYPE" and "CLASSIFICATION" had binning applied before OneHotEncoder was used to transform the categorical values.  The dataset was then split for training and testing sets and scaled using StandardScaler.  
    

## Compiling, Training, and Evaluating the Model
For the initial deep-learning neural network model, the input data had 43 features.  Three layers total were used.  The first two were hidden layers using the "relu" activation function and had neurons of 80 and 30 and respectively.  The output model used the sigmoid function since our target output is binary.  With these parameters in place, the evaluation of the model only had 73.08% accuracy, falling below our target model performance.  

![Initial Model](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/Initial_Model.png) 
  

## Optimizing the Model

### Attempt 1
The first and second hidden layers' activation functions were changed to tanh to see if the model could be brought above the 75% target model performance.  It performed slightly worse than the initial model at a 72.89% accuracy on the test data.


![First Attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/firstattempt.png) 
  
 
### Attempt 2
The activation functions in the first and second hidden layers were changed back to "relu".  A third hidden layer was added that also used the "relu" activation function.  Then the number of neurons were increase for the first and second hidden layer to 100 and 40 while the third hidden layer was set to 20.  These parameters performed roughly the same at a 72.84% accuracy on the test data, again falling below the 75% threshold goal.  


![Second attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/secondattempt.png) 
 

### Attempt 3
The binning process for the "APPLICATION_TYPE" and "CLASSIFICATION" columns done in the initial model was altered slightly for the third attempt at optimizing the model.  "APPLICATION_TYPE" increased from 9 bins to 10 bins while "CLASSIFICATION" increased from 6 bins to 9 bins.  In addition, the # of neurons in the three hidden layers was decreased to 30, 15, and 5.  This attempt, again fell below our target 75% and performed similar to the other attempts with an accuracy of 72.87% on the test data.  


![Third Attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/thirdattempt.png) 
 


## Summary
The deep learning neuron network model was unable to be optimized for a 75% accuracy performance through the intial attempt and the 3 addtional attempts. During the 4 different attempts the activation functions were changed and number of neurons in the hidden layers, adding a third hidden layer, and altering the binning process to some degree on two columns that required this preprocessing step. For future attempts at optimizing this model, it would be a good idea to apply binning to the "ASK_AMOUNT" column. Another alternative would be using a random forest classifier model on the data to see if better results are acheived.  Random forest classifiers will allow for a classification output and easily handle outliers and nonlinear data.  
