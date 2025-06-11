# -*- coding: utf-8 -*-
"""
This module implement a deep learning model using the parameters which get the 
best accuracy in the first deep learning model
"""
# Modulos para redes neuronales
from keras.models import Sequential
from keras.layers import Dense, Input
import pandas as pd   
import matplotlib.pyplot as plt   
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def modelo2_best_grid(x_train, y_train):
    def _create_model(neuronasEntrada):    
        modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(96, activation="relu"),
            Dense(1, activation='sigmoid')
            ])
        #  Compilacion
        modelo.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return modelo

    # x_train.shape[1] es el numero de neuronas de entrada 
    modelo = _create_model(x_train.shape[1])
    
    # Veamos los detalles del modelo
    modelo.summary()

    # Entrenamiento del modelo
    history = modelo.fit(x_train, 
                         y_train, 
                         epochs = 40, 
                         batch_size = 64,
                         validation_split = 0.2
                         )  

    # Visualizando el resultado
    print(history.history.keys())
    df = pd.DataFrame(history.history)
    print(df)
    df.plot()

    # Representamos por separado la evolucion de la funcion de perdida y el accuracy
    dfAccuracy = df.loc[:,["accuracy","val_accuracy"]]
    dfAccuracy.plot()

    dfLoss = df.loc[:,["loss","val_loss"]]
    dfLoss.plot()

    # Prediccion
    predicciones = modelo.predict(x_train) #sigmoidal devuelve la probabilidad de que sea 1
    print(predicciones[0:10,])
    # Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
    predic_train = 1*(predicciones>0.5)
    print(predic_train[0:10,])

    # Datos reales
    print(y_train[0:10,])

    # Matriz de contingencia o matriz de confusion
    mc = confusion_matrix(y_train, predic_train)
    print(mc)

    # el modelo solamente dice que no para todos los casos, que nadie sufrirá ictus
    # Graficamos la matriz
    class_names = ['NoChurn', 'Churn']
    disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
    disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
    
    # Salvando el modelo
    modelo.save('alternative2.keras')
