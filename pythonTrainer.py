"""
Created on Tue Sep 17 22:40:02 2019
@author: Esmond Dsouza
"""

import pandas as pd
import numpy as np

mainData = pd.read_csv("C:/Users/Esmond Dsouza/Desktop/Machine Learning/tcdMainData.csv")
predictionData = pd.read_csv("C:/Users/Esmond Dsouza/Desktop/Machine Learning/tcd ml 2019-20 income prediction test (without labels).csv")
predictionData['Income'] = predictionData['Income'].replace(np.nan, 0)
mainData.rename(columns={'Income in EUR':'Income'}, inplace=True)

#dropping al rows that have 3 or more NaN values

indexOfNaNRows = []
for i in range(len(mainData.index)):
    NaNCount = mainData.iloc[i].isnull().sum()
    if NaNCount >= 3:
        indexOfNaNRows.append(i)
mainData.drop(index=indexOfNaNRows, axis=0, inplace=True)

#concatenating the training and prediction data 
#mainData.fillna(method = 'bfill', inplace=True)
#mainData.dropna(inplace=True)
combinedData = pd.concat([mainData, predictionData])

#converting the string values to numbers
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
combinedData['Country'] = combinedData['Country'].replace(['unknown','0',np.nan],'unknown')
combinedData['Profession'] = combinedData['Profession'].replace(['unknown','0',np.nan],'unknown')
combinedData['University Degree'] = combinedData['University Degree'].replace(['unknown','0',np.nan],'unknown')
#combinedData['Year of Record'] = combinedData['Year of Record'].replace([np.nan],combinedData['Year of Record'].mean())

#combinedData['Country'] = le.fit_transform(combinedData['Country'])
#combinedData['Gender'] = combinedData['Gender'].apply(lambda x: 'other' if x !='male' and x !='female' else x)
#combinedData['Gender'] = le.fit_transform(combinedData['Gender'])
#combinedData['University Degree'] = combinedData['University Degree'].apply(lambda x: 'No' if x !='Master' and x !='Bachelor' and x != 'PhD' else x)
#combinedData['University Degree'] = le.fit_transform(combinedData['University Degree'])
#combinedData['Profession'] = combinedData['Profession'].replace(np.nan, 'other')
#combinedData['Profession'] = le.fit_transform(combinedData['Profession'])
#combinedData['Hair Color'] = combinedData['Hair Color'].apply(lambda x: 'Other' if x !='Black' and x !='Brown' and x != 'Blonde' and x != 'Red' else x)
#combinedData['Hair Color'] = le.fit_transform(combinedData['Hair Color'])
#combinedData['Size of City'] = le.fit_transform(combinedData['Size of City'])

#binning process for integer columns 
ageBins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ageBinLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
combinedData['Age'] = pd.cut(combinedData['Age'], bins=ageBins, labels=ageBinLabels)
yearBins = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
yearBinLables = ['80-85', '85-90', '90-95', '95-00', '00-05', '05-10', '10-15', '15-20']
combinedData['Year of Record'] = pd.cut(combinedData['Year of Record'], bins=yearBins, labels=yearBinLables)
heightBins = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280]
heightBinLabels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
combinedData['Body Height [cm]'] = pd.cut(combinedData['Body Height [cm]'], bins=yearBins, labels=yearBinLables)

#separating the training and prediction data after pre processing
combinedData = pd.get_dummies(combinedData, columns =['Gender', 'Country', 'Profession', 'Hair Color', 'University Degree', 'Age', 'Year of Record'])
combinedData.drop(columns =['Body Height [cm]', 'Size of City'], inplace= True)
transformedMainData = combinedData[combinedData.Income != 0]
transformedPredictionData = combinedData[combinedData.Income == 0]

#preparing the data for test splitting
X_transformedMainData = transformedMainData.drop(['Income', 'Instance'], axis=1)
Y_transformedMainData = transformedMainData['Income']
X_transformedPredictionData = transformedPredictionData.drop(['Income'], axis=1)


#splitting sample data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_transformedMainData, Y_transformedMainData, test_size = 0.20)
from sklearn import metrics

"""
from sklearn import linear_model
trainingRegressor = linear_model.BayesianRidge(alpha_1=1e-07, alpha_2=1e-07, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-07, lambda_2=1e-07, n_iter=300,
       normalize=True, tol=0.001, verbose=True)
trainingRegressor.fit(X_train, Y_train)
Y_pred = trainingRegressor.predict(X_test)
df = pd.DataFrame({'Test': Y_test, 'Prediction': Y_pred})
print('Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
"""
from xgboost import XGBRegressor
trainingRegressor = XGBRegressor(n_estimators = 40, max_depth = 3, learning_rate = 0.95, subsample = 0.25)
trainingRegressor.fit(X_train, Y_train)
Y_pred = trainingRegressor.predict(X_test)
df = pd.DataFrame({'Test': Y_test, 'Prediction': Y_pred})
print('Root Mean Squared Error', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    
#plotting the graph for the first 25 values
import matplotlib.pyplot as plt  
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#training model with all the data
regressor = linear_model.BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=True, tol=0.001, verbose=True)  
#X_transformedMainData = sc_X.fit_transform(X_transformedMainData)
regressor.fit(X_transformedMainData, Y_transformedMainData)

#predicting the actual data
Y_trasnformedPredictionData = regressor.predict(X_transformedPredictionData.drop(['Instance'], axis=1));
finalPredictionSheet = pd.DataFrame({'Instance': X_transformedPredictionData['Instance'], 'Income': Y_trasnformedPredictionData})
finalPredictionSheet.to_csv(r'C:/Users/Esmond Dsouza/Desktop/Machine Learning/prediction1234.csv', index = None, header=True)