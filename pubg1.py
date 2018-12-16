import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

dataset  = pd.read_csv('train.csv')
dataset2  = pd.read_csv('test.csv')
predcsv = pd.DataFrame({'Id':dataset2['Id']})

##checking for the NA values
dataset.isnull().sum().sort_values()
dataset2.isnull().sum().sort_values()

dataset.drop(['Id','groupId', 'matchId','numGroups','teamKills','longestKill'],axis = 1, inplace = True)
dataset2.drop(['Id','groupId', 'matchId','numGroups','teamKills','longestKill'],axis = 1, inplace = True)


X = dataset.drop(['winPlacePerc'],axis = 1)
y = dataset['winPlacePerc']
X_final = dataset2


#--------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_final = sc.transform(X_final)   
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


plt.scatter(dataset['walkDistance'],dataset['winPlacePerc'],c = 'red')


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 19))
classifier.add(Dropout(0.5))
#Adding the second hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 5)


y_pred = classifier.predict(X_test)


predcsv['winPlacePerc'] = y_pred
predcsv.to_csv('sample_submission.csv',index=False)