# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:10:52 2021

@author: valentin
"""
#####Extraction de données historiques sur les cours des actions
import pandas as pd
import numpy as np
 
# Pour supprimer la notation scientifique des tableaux numpy
np.set_printoptions(suppress=True)
 
#nsepy pour obtenir les prix des actions
!pip install nsepy
 
#Permet d'obtenir des données boursières grâce à nsepy
from nsepy import get_history
from datetime import datetime
 
startDate=datetime(2019, 1,1)
endDate=datetime(2020, 10, 5)
 
# Récupérer les données
StockData=get_history(symbol='INFY', start=startDate, end=endDate)
print(StockData.shape)
StockData.head()


##### Représentation graphique des data

StockData['TradeDate']=StockData.index
 

%matplotlib inline
#abcisse -> date et ordonné -> prix
StockData.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)


####Mise en forme, préparation des données pour LSTM

# Récupération des prix journaliers
FullData=StockData[['Close']].values
print(FullData[0:5])
 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
 
# normalisation des data
sc=MinMaxScaler()
 
DataScaler = sc.fit(FullData)
X=DataScaler.transform(FullData)
 
print('apres la normalisation')
X[0:5]

# Matrices d'entrainement
X_samples = []
y_samples = []

NumerOfRows = len(X)\n
TimeSteps=10  # dernier jour de la matrice entrainement,

# Mise en place, préparation pour l'apprentissage renforcé
for i in range(TimeSteps , NumerOfRows , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)

# Reshape de la matrice en matrice 3 dimensions (number of samples, Time Steps, Features)
X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
print('Entrée')
print(X_data.shape)

y_data=np.array(y_samples)
y_data=y_data.reshape(y_data.shape[0], 1)
print('sortie')
print(y_data.shape)


#####Préparation des matrices d'entrainelent et de test

##Nb d'archives testées
TestingRecords=5

# Répartition matrice entrain,ement et test
X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]

# Vérification des dimensions des matrices
print('\\n Training Data shape')
print(X_train.shape)
print(y_train.shape)
print('\\n Testing Data shape ')
print(X_test.shape)
print(y_test.shape)



#####Permet de visualiser ce qu'on envoie au LSTM
for inp, out in zip(X_train[0:2], y_train[0:2]):
    print(inp,'--', out)


###### Création du model LSTM
	
# Choix dimension d'netrée
TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
print("Number of TimeSteps:", TimeSteps)
print("Number of Features:", TotalFeatures)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# initialisation RNN
regressor = Sequential()
 
# 1er couche
regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
 
# 2eme couche
regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
 
# 3eme couche
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
 
 
# sortie
regressor.add(Dense(units = 1))
 
# lancement RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
 
 
import time
##Ajout d'un timer
StartTime=time.time()
 
# lancement RNN avec l amatrice d'entrainement
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
 
EndTime=time.time()
print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')


######Test fiabilité
	
# Prédiction sur la matrice de test
predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
 
# prix d'origine
orig=y_test
orig=DataScaler.inverse_transform(y_test)
 
# fiabilité de la prediction
print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())
 
# plot du resultat
import matplotlib.pyplot as plt
 
plt.plot(predicted_Price, color = 'blue', label = 'Predicted Volume')
plt.plot(orig, color = 'lightblue', label = 'Original Volume')
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(TestingRecords), StockData.tail(TestingRecords)['TradeDate'])
plt.ylabel('Stock Price')
 
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()





###Prédiction sur tout le set de data
TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))
 
FullDataPredictions=np.append(TrainPredictions, TestPredictions)
FullDataOrig=FullData[TimeSteps:]
 
# plot de toutes les data
plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
plt.plot(FullDataOrig , color = 'lightblue', label = 'Original Price')
 
 
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()




#####Prédiction pour le du prix du lendemain

Last10Days=np.array([1002.15, 1009.9, 1007.5, 1019.75, 975.4,
            1011.45, 1010.4, 1009,1008.25, 1017.65])
 
# Normalisatiopn des data
Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))
 
# reshape des matrices
NumSamples=1
TimeSteps=10
NumFeatures=1
Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

 
#Prédictions
predicted_Price = regressor.predict(Last10Days)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
predicted_Price







