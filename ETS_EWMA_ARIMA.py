# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:58:52 2021

@author: NGUEND
"""

"""
Une grande partie des informations en finance se representent sous la forme de
valeurs tracées en fonction d'une série temporelle

Le but des algorithmes que nous allons aborder dans ce dossier, sont là pour
montrer pourquoi l'utilisation de certaines techniques de ces analyses sur des
infos boursières ne sont pas une bonne idée.
Bien qu'elles donnent envie de les utiliser il faut garde en tête que ce n'est
pas toujours la meilleure approche (notamment ARIMA).

Le but de notre étude est d'utiliser les série temporelles au sein de données
financières, par ailleurs les méthodes que nous allons voir sont très efficaces 
pour d'autres usages.'

Voici le plan de nos recherches :
    - Notions de base sur les séries temps
    - Library Statsmodels 
    - Modèle ETS (erreur, tendance, saisonnalité) et leur décomposition
    - Modèle EWMA (moyenne mobile pondéré exponentiellement)
    - Modèle ARIMA (moyenne mobile autoregressif intégré)


NB : pour les différents fichiers nous utlisons 2 csv que vous trouverez 
dans le dossier.
"""

"""BASE DES SERIES TEMPS

- Tendances : 
    Image1
    Une tendance décrit en moyenne l'évolution de la valeur pour cette série
    temporelle.
    Comme on voit sur le graphe 3 type de tendances.

- Saisonnalité :
    Image2 (tendances google été)
    Tendance qui se répète comme les saisons, sur le graphe on voit bien qu'il 
    y a une tendance d'été pendant les étés
    
- Cyclique :
    Image3
    Tendance sans répitition définie. On ne peut donc pas définir la saison sur
    ce genre de graphe, d'où les cycles.
    
"""


"""INTRODUCTION STATSMODEL

Module python qui permet d'explorer des données, d'estimer des modèles
statistiques et d'effectuer des tests statistiques.

pour installer :
    conda install statsmodels

"""
""" 1ER EXEMPLE """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def exemple1():

    df = sm.datasets.macrodata.load_pandas().data
    """ Un tableau rassemblant des données basique du gov américain,
    pour plus d'explication print le rapport détaillé ci-dessous"""
    # print(sm.datasets.macrodata.NOTE)
    
    index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1','2009Q3'))
    df.index = index
    """ Index datetime trimestrielle """
    
    df['realgdp'].plot()
    """ Evolution du gdp """
    
    gdp_cycle,gdp_trend = sm.tsa.filters.hpfilter(df['realgdp']) #tuple de 2 el.
    
    df['trend'] = gdp_trend
    df[['realgdp','trend']]['2000-03-31':].plot()
    """ On peut donc voir la tendance à partir des valeurs réelles
    on pu utiliser des formules de mathématiques puissante en 5 lignes"""


""" 2EME EXEMPLE """

def exemple2():
    
    """MODELE ETS : ERREUR TENDANCE SAISON 
    Prennent chacun de ces termes pour le "lissage" et peuvent ajouter,
    multiplier,  ou même en omettre.
    
    Sur la base de ces facteurs clés, nous pouvons essayer de créer un modèles 
    adapter à nos données
    """
        
    airline = pd.read_csv('airline_passengers.csv', index_col = 'Month')
    # tendance linéaire ou exp ?
    
    airline.dropna(inplace = True)
    
    airline.index = pd.to_datetime(airline.index)
    
    result = seasonal_decompose(airline['Thousands of Passengers'], model = 'multiplicative')
    result.seasonal.plot()
    result.trend.plot()
    result.plot()
    """ model peut etre additive ou multiplicative. 
    Il suffit de test les deux pour voir lequel correspond le mieux """

exemple2()

""" 3EME EXEMPLE """
def exemple3():
    
    """MODEL EWMA (Moyennes mobiles pondérées exponentiellement)
    DECRIT LES TENDANCES DES DONNEES
    Les moyennes mobiles permettent de créer des modèles simple
    qui décrivent une tendance générale d'une série temporelle.
    Dans le graphe ci-dessous on utilise la moyenne mobile simple (12 et 6 mois)
    Image4
    
    Quelques faiblesses dans la méthode SMA (simple), les périodes sont plus petites donc
    entraîne plus de bruit que de signal. TOujours un décalage donc un manque de données.
    De plus il n'atteindra jamais le sommet ou la vallée complète des données dû
    à la moyenne. Et il n'informe pas sur le futur juste des tendances.
    
    EWMA permet de réduire le décalage et accorde + de poids aux valeurs récentes.
    Le poids dépends de paramètres et et nombres de périodes.
    """

    airline = pd.read_csv('airline_passengers.csv', index_col ='Month')
    airline.dropna(inplace= True)
    airline.index = pd.to_datetime(airline.index)
    
    airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 6).mean()
    
    airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 12).mean()
    
    airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12).mean()

    airline[['Thousands of Passengers','EWMA12']].plot()



"""4EME EXEMPLE"""

def exemple4():
    """ ARIMA ne fonctionnne pas bien pour les données financières historiques
    Mais pour la prévision de vente dans une entreprise c est parfait.
    Donc on utilise cette méthode pour la prévision.
    
    Il y a 2 types : Saisonnier et non-saisonnier
    On utilise ARIMA lorsque les données montrent des signe de non-stationnarité.
    ARIMA dépend de p,d,q des entiers non négatifs.
    p: Autorégression
        Relation de dépendance entre les observations actueles et antérieure.
    
    d: Intégré
        Différencier les observations pour rendre le tout stationnaire.
        
    q: Moyenne mobile
        Modèle qui utilise la dépendances entre observation et erreur résiduelle d'un modèle
        de moyenne mobile appliqué aux observations décalées.
        
    Pour étudier la stationnarité on utilise le test Augmented Dickey-Fuller.
    Voir plus bas.
    
    Utilisons les graph d'autocorrélation pour montrer la corrélation de la série 
    avec elle-même décalée de X unités de temps.
    
    
    Les étapes du modèles sont:
        Visualiser les données de la série temporelle
        Rendre les données de la série temporelle stationnaires
        Tracer les graph d'auto ou de corrélation
        Construire le modèle ARIMA
        Faire des prédictions
        """
        
    
    # Visualisation des données

    df = pd.read_csv('monthly-milk-production-pounds-p.csv')
    
    df.columns = ['Month', 'Litre de lait par vache']
    df.drop(168, axis =0, inplace = True)
    
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace = True)
    
    desc = df.describe().transpose()
    #Pour avoir des données statistiques sur les litres de vaches
    
    time_series = df['Litre de lait par vache']
    # time_series.rolling(12).mean().plot(label = 'Moyenne mob sur 12')
    # time_series.plot(label = 'lait par vache')
    # time_series.rolling(12).std().plot(label= 'ecart type')
    # plt.legend()
    
    decomposition = seasonal_decompose(time_series)
    # fig = decomposition.plot()
    #Pour voir la tendance la saisonnalité et l'erreur résiduelle
    # fig.set_size_inches(15,8)
    
    
    """ Nous voulons connaître si c'est stationnaire ou pas"""    
    
    result = adfuller(df['Litre de lait par vache'])
    
    def adf_check(time_series):
        result = adfuller(time_series)
        print('test de Dick fuller augmente DAF')
        labels = ['ADF Test statitstic', 'p-value','Lags used', 'number of observation']
        
        for value, label in zip(result, labels):
            print(label + ': ' + str(value))
            
        if result[1] <= 0.05:
            print("forte preuve contre l hypo null")
            print("rejette l hypo null")
            print("les data n ont pas de racine unitaire et sont donc statio")
        else:
            print("faible evidence contre l hypo nulle")
            print("echoue a rejeter l hypo null")
            print("les data ont une racine unitaire, elles sont donc no station")
            
    # adf_check(df['Litre de lait par vache'])
    '''test de Dick fuller augmente DAF
    ADF Test statitstic: -1.3038115874221226
    p-value: 0.6274267086030347
    Lags used: 13
    number of observation: 154
    faible evidence contre l hypo nulle
    echoue a rejeter l hypo null
    les data ont une racine unitaire, elles sont donc no station
    '''
    # on constate bien via la decomposition saison que ce n'etait pas station
    
    df['First Difference'] = df['Litre de lait par vache'] - df['Litre de lait par vache'].shift(1)
    # df['First Difference'].plot()
    
    # adf_check(df['First Difference'].dropna())
    # ici station
    # mais si tjrs pas statio on continue de le faire
    
    df['Second diff'] = df['First Difference'] - df['First Difference'].shift(1)
    # adf_check(df['Second diff'].dropna())
    # df['Second diff'].plot()
    
    # Diff season
    
    df['Seasonal diff'] = df['Litre de lait par vache'] - df['Litre de lait par vache'].shift(12)
    # df['Seasonal diff'].plot()
    
    # adf_check(df['Seasonal diff'].dropna())
    '''
    test de Dick fuller augmente DAF
    ADF Test statitstic: -2.3354193143593998
    p-value: 0.16079880527711288
    Lags used: 12
    number of observation: 143
    faible evidence contre l hypo nulle
    echoue a rejeter l hypo null
    les data ont une racine unitaire, elles sont donc no station
    '''
    
    df['Seasonal first diff'] = df['First Difference'] - df['First Difference'].shift(12)
    df['Seasonal first diff'].plot()
    adf_check(df['Seasonal first diff'].dropna())
    
    '''
    test de Dick fuller augmente DAF
    ADF Test statitstic: -5.0380022749219915
    p-value: 1.865423431878764e-05
    Lags used: 11
    number of observation: 143
    forte preuve contre l hypo null
    rejette l hypo null
    les data n ont pas de racine unitaire et sont donc statio
    '''
    
    """Courbe autocorel et autocorel partielle """
    
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig_first = plot_acf(df['First Difference'].dropna())
    fig_season_first = plot_acf(df['Seasonal first diff'].dropna())
    
    res = plot_pacf(df['Seasonal first diff'].dropna())
    
    fig = plot_acf(df['Seasonal first diff'].dropna())
    fig = plot_pacf(df['Seasonal first diff'].dropna())
    
    # Utiliser ARIMA pour prédir donnée futur
    
    from statsmodels.tsa.arima_model import ARIMA
    model = sm.tsa.statespace.SARIMAX(df['Litre de lait par vache'], order =[0,1,0], seasonal_order = (1,1,1,12))
    
    res = model.fit()
    print(res.summary())
    res.resid.plot(kind = 'kde')
    
    df['forecast'] = res.predict(start = 150, end = 200)
    df[['Litre de lait par vache','forecast']].plot(figsize = (12,8))
    
    '''
    on vient de faire une prévision des valeurs apres avoir étudié la stationnarité de nos data
    on trouve qqch de cohérent sur les dernieres valeur, on a donc reussi à estimer les 
    litres de lait par vache sur 1 année
    '''
    # essayons de rajouter des prédictions pour plus loin donc ajouter de nouvelles lignes de date
    
    from pandas.tseries.offsets import DateOffset
    
    futur_date = [df.index[-1] + DateOffset(months = x) for x in range(1,24)]
    
    futur_df = pd.DataFrame(index = futur_date, columns = df.columns)
    final_df_futur = pd.concat([df,futur_df])
    final_df_futur['forecast'] = res.predict(start = 168, end = 192 )
    final_df_futur[['Litre de lait par vache','forecast']].plot(figsize = (12,8))
        
            
        
"""
Pour conclure,
Dans les données fiancières, trop de facteurs exterieurs au temps.
On dit que le prix des actions suivent des mouvements browniens (aléatoires),
donc grosse fluctuaction.
    
    
    
