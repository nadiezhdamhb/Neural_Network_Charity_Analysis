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
  3.  The remaining columns will be used for the features of the machine learning model.  "APPLICATION_TYPE" and "CLASSIFICATION" had binning applied before OneHotEncoder was used to transform the categorical values. The dataset was then split for training and testing sets and scaled using StandardScaler.  
    

## Compiling, Training, and Evaluating the Model
For the initial deep-learning neural network model with 43 features, three layers were used. The first two were hidden layers using the "relu" activation function had neurons of 80 and 30 respectively. Since the target output is binary, I used the sigmoid function however, the model only had 73.08% accuracy which is a little bit lower than the target model performance of 75%.  

![Initial Model](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/Initial_Model.png) 
  

## Optimizing the Model

### Attempt 1
For the first attempt, I tried changing the activation functions on the the first and second hidden layers to tanh, unfortunatley it resulted in a slightly lower accuracy than the initial model at a 72.89% accuracy on the test data.


![First Attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/firstattempt.png) 
  
 
### Attempt 2
For the second attempt, the activation functions in the first and second hidden layers were changed back to "relu" and tried adding a third hidden layer with 20 neurons with the same "relu" activation function. I also added more neurons for the first and second hidden layer to 100 and 40 respectively. Unfortunately, it still resulted on about the same 72.84% accuracy on the test data which still is less than 75% goal.  


![Second attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/secondattempt.png) 
 

### Attempt 3
For the third attempt, I tried changing a little bit the binning process for the "APPLICATION_TYPE" and "CLASSIFICATION" columns from what it was done in the initial attempt.  Moreover, "APPLICATION_TYPE" increased from 9 bins to 10 bins while "CLASSIFICATION" increased from 6 bins to 9 bins and more neurons were added to the three hidden layers. Once again, the model resulted on an accuracy of 72.87% which is below 75%.  


![Third Attempt](https://github.com/nadiezhdamhb/Neural_Network_Charity_Analysis/blob/main/Resources/Images%20for%20Readme/thirdattempt.png) 
 


## Summary
The deep learning neuron network model was unable to be optimized for a 75% accuracy performance through the intial attempt and the 3 additional attempts. During the 4 different attempts the activation functions were changed and number of neurons in the hidden layers, adding a third hidden layer, and altering the binning process to some degree on two columns. For future attempts at optimizing this model, it would be a good idea to apply binning to the "ASK_AMOUNT" column. Another alternative would be using a random forest classifier model on the data to see if better results are acheived. I believe the Random forest classifiers will be useful for a classification output, easy handle of outliers and nonlinear data.  
