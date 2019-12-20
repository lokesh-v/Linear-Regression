# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import seaborn as sns 

# Importing the dataset

data=pd.read_csv("auto_mpg_dataset_1 (1).csv")

num_col=data._get_numeric_data().columns
len_num_col=len(num_col)

#fig, axes = plt.subplots(1, 5, figsize=(10,2.5), dpi=100, sharex=True, sharey=True)
#EDA    

#Histagrams
plt.hist(data["cylinders"]) #cat
plt.gca().set(title='Cylinder Histogram', ylabel='Frequency')

plt.hist(data["displacement"])
plt.gca().set(title='Displacement Histogram', ylabel='Frequency')

plt.hist(data["horsepower"]) 
plt.gca().set(title='Horse Power Histogram', ylabel='Frequency')

plt.hist(data["weight"])
plt.gca().set(title='Weight Histogram', ylabel='Frequency')

plt.hist(data["acceleration"]) 
plt.gca().set(title='Acceleration Histogram', ylabel='Frequency')

plt.hist(data["model_year"]) 
plt.gca().set(title='Model_Year Histogram', ylabel='Frequency')

plt.hist(data["origin"]) #cat
plt.gca().set(title='Origin Histogram', ylabel='Frequency')

plt.hist(data["mpg"])
plt.gca().set(title='MPG Histogram', ylabel='Frequency')

#Density Plots for Normality

#scatter plot
plt.scatter(data['mpg'],data['acceleration'])

sns.kdeplot(data["displacement"])
sns.kdeplot(data["weight"])
sns.kdeplot(data["acceleration"])
sns.kdeplot(data["model_year"])
sns.kdeplot(data["mpg"])

data.isnull().sum()
data_EDA=data.copy()
data_EDA.isnull().sum()


data_EDA['displacement'] = data_EDA['displacement'].fillna((data_EDA['displacement'].median()))  
data_EDA['weight'] = data_EDA['weight'].fillna((data_EDA['weight'].median())) 
data_EDA['horsepower'] = data_EDA['horsepower'].fillna((data_EDA['horsepower'].median()))
mode=data_EDA['cylinders'].mode() 
type(mode) 
data_EDA['cylinders'] = data_EDA['cylinders'].fillna(mode[0])

data_EDA.isnull().sum()

plt.boxplot(data_EDA['displacement'])
plt.boxplot(data_EDA['weight'])
plt.boxplot(data_EDA['horsepower'])
plt.boxplot(data_EDA['cylinders'])
plt.boxplot(data_EDA['acceleration'])
plt.boxplot(data_EDA['model_year'])
plt.boxplot(data_EDA['origin'])
plt.boxplot(data_EDA['mpg'])

data_EDA.isnull().sum()

# Outliers treatment
percentiles = data_EDA['displacement'].quantile([0.1,0.9]).values
data_EDA['displacement']=data_EDA['displacement'].clip(percentiles[0],percentiles[1])
percentiles = data_EDA['horsepower'].quantile([0.1,0.9]).values
data_EDA['horsepower']=data_EDA['horsepower'].clip(percentiles[0],percentiles[1])
percentiles = data_EDA['acceleration'].quantile([0.1,0.9]).values
data_EDA['acceleration']=data_EDA['acceleration'].clip(percentiles[0],percentiles[1])
percentiles = data_EDA['mpg'].quantile([0.1,0.9]).values
data_EDA['mpg']=data_EDA['mpg'].clip(percentiles[0],percentiles[1])

plt.boxplot(data_EDA['displacement'])
plt.boxplot(data_EDA['weight'])
plt.boxplot(data_EDA['horsepower'])
plt.boxplot(data_EDA['cylinders'])
plt.boxplot(data_EDA['acceleration'])
plt.boxplot(data_EDA['model_year'])
plt.boxplot(data_EDA['origin'])
plt.boxplot(data_EDA['mpg'])

data_EDA.columns
del data_EDA["car_name"]
data_EDA.columns

#One hot encoding

dfDummies_cyl = pd.get_dummies(data_EDA['cylinders'], prefix = 'Cyl')
dfDummies_orig = pd.get_dummies(data_EDA['origin'], prefix = 'orig')

data_OHC = pd.concat([data_EDA, dfDummies_cyl,dfDummies_orig], axis=1)

del data_OHC['cylinders']
del data_OHC['origin']



#Defining Target and Predictors

y_OHC=data_OHC["mpg"]
del data_OHC["mpg"]
X_OHC=data_OHC.copy()


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_OHC_train, X_OHC_test, y_OHC_train, y_OHC_test = train_test_split(X_OHC, y_OHC, test_size = 0.2, random_state = 0)

# using Stats model
import statsmodels.api as sm
# Adding constant
X_OHC_train_sm = sm.add_constant(X_OHC_train)

X_OHC_test_sm = sm.add_constant(X_OHC_test)



from sklearn.metrics import r2_score

model_OHC = sm.OLS(y_OHC_train, X_OHC_train_sm).fit()
predictions_OHC = model_OHC.predict(X_OHC_test_sm) # make the predictions by the model
r2_OHC=r2_score(y_OHC_test,predictions_OHC)


# Print out the statistics
model_OHC.summary()


data_OHC.columns

data_OHC_DVT=data_OHC.copy()

del data_OHC_DVT['Cyl_3.0']
del data_OHC_DVT['orig_1']

y_OHC_DVT=y_OHC.copy()
X_OHC_DVT=data_OHC_DVT
X_OHC_DVT = sm.add_constant(X_OHC_DVT)

X_OHC_DVT_train, X_OHC_DVT_test, y_OHC_DVT_train, y_OHC_DVT_test = train_test_split(X_OHC_DVT, y_OHC_DVT, test_size = 0.2, random_state = 0)

model_OHC_DVT = sm.OLS(y_OHC_DVT_train, X_OHC_DVT_train).fit()
predictions_OHC_DVT = model_OHC_DVT.predict(X_OHC_DVT_test) # make the predictions by the model
r2_OHC_DVT=r2_score(y_OHC_DVT_test,predictions_OHC_DVT)

model_OHC_DVT.summary()

# Deleting insignificant features
del data_OHC['displacement']
del data_OHC['horsepower']
del data_OHC['acceleration']
del data_OHC['Cyl_4.0']
del data_OHC['Cyl_5.0']
del data_OHC['Cyl_8.0']
del data_OHC['orig_3']

X_OHC=data_OHC.copy()

from sklearn.model_selection import train_test_split
X_OHC_train, X_OHC_test, y_OHC_train, y_OHC_test = train_test_split(X_OHC, y_OHC, test_size = 0.2, random_state = 0)

X_OHC_train_sm = sm.add_constant(X_OHC_train)

X_OHC_test_sm = sm.add_constant(X_OHC_test)

model_OHC = sm.OLS(y_OHC_train, X_OHC_train_sm).fit()
predictions_OHC = model_OHC.predict(X_OHC_test_sm) # make the predictions by the model
r2_OHC=r2_score(y_OHC_test,predictions_OHC)


# Print out the statistics
model_OHC.summary()

















