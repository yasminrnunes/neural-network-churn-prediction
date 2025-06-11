# -*- coding: utf-8 -*-
"""
This modules processes the data and adjust it to be used in a deep learning model
"""
# Cargar paquetes
import numpy as np      
import pandas as pd     
#import matplotlib.pyplot as plt 
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def processing_data(X,y):
  # Verificando el formato de los datos
  X.shape
  X.head(10)
  
  # Verificando los registros faltantes
  X.isna().sum()
  
  # Verificar si los datos están bien distribuidos
  np.unique(y, return_counts=True)[1]/y.shape[0]
  
  # Uniendo los datos
  data = X.copy()
  data['y'] = y
  
  # Redistribuicion de los datos
  filtro = data['y'] == 1
  
  # Valores originales
  print("Originales: ",sum(filtro),"\n")
  nuevos = data[filtro].sample(n=2100, replace=True) # se añaden 2100 datos que es churn 1
  
  # Datos ampliados
  data_ampliado = pd.concat([data, nuevos])
  data_ampliado.shape
  
  # Generamos de nuevo las divisones de entrenamiento y prueba
  # y hacemos el tratamiento de variables numericas y categoricas
  X = data_ampliado.iloc[:, :-1]
  y = data_ampliado.iloc[:, -1]
  
  # Verificar distribuicion de los datos
  np.unique(y, return_counts=True)[1]/y.shape[0]
  
  # Separar los datos en train y test
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y) #stratify es importante utilizar para problemas
  # Proporcion de las clases en los diferentes conjuntos
  np.unique(y, return_counts=True)[1]/y.shape[0]
  np.unique(y_train, return_counts=True)[1]/y_train.shape[0]
  np.unique(y_test, return_counts=True)[1]/y_test.shape[0]
  
  # Ajustando los campos
  X.dtypes
  X.shape #importante ejecutar para confirmar que sean object e int
  print(X['Call  Failure'].unique()) #numerica
  print(X['Complains'].unique()) # categorica
  print(X['Subscription  Length'].unique()) #numerica
  print(X['Charge  Amount'].unique()) # categorica
  print(X['Seconds of Use'].unique()) #numerica
  print(X['Frequency of use'].unique()) #numerica
  print(X['Frequency of SMS'].unique()) #numerica
  print(X['Distinct Called Numbers'].unique()) #numerica
  print(X['Age Group'].unique()) # categorica
  print(X['Tariff Plan'].unique()) # categorica
  print(X['Status'].unique()) # categorica
  print(X['Age'].unique()) #numerica
  print(X['Customer Value'].unique()) #numerica
  
  # Definiendo las variables numéricas
  num_var = ['Call  Failure','Subscription  Length','Seconds of Use','Frequency of use',
            'Frequency of SMS','Distinct Called Numbers','Age','Customer Value']
  
  # Definiendo las variables categóricas
  cat_var = ['Complains','Charge  Amount','Age Group','Tariff Plan','Status']
  
  # Nombre de todas las variables
  #columns = ['Call  Failure','Complains','Subscription  Length','Charge  Amount','Seconds of Use','Frequency of use',
  #        'Frequency of SMS','Distinct Called Numbers','Age Group','Tariff Plan','Status']
    
  # Aplicando las transformaciones
  transformer = make_column_transformer(
        (StandardScaler(), num_var),
        (OneHotEncoder(), cat_var),
        verbose_feature_names_out=False)
    
  # Ajustamos utilizando informacion del conjunto de entrenamiento
  transformer.fit(x_train)
    
  # Aplicando la transformacion a los datos de entrenamiento
  x_train = transformer.transform(x_train)
  x_train = pd.DataFrame(x_train, columns=transformer.get_feature_names_out())
  # Aplicando la transformacion a los datos de test
  x_test = transformer.transform(x_test)
  x_test = pd.DataFrame(x_test, columns=transformer.get_feature_names_out())
  
  return x_train,x_test,y_train,y_test


