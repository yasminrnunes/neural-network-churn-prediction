# -*- coding: utf-8 -*-
"""
This module implement a deep learning model using the parameters which get the 
best accuracy in the first deep learning model
"""
     
import pandas as pd 
import matplotlib.pyplot as plt  
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def modelo3_exponential_decay(x_train, y_train):
    def _create_model(neuronasEntrada):    
        modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(96, activation="relu"),
            Dense(1, activation='sigmoid')
            ])
        return modelo

    # x_train.shape[1] es el numero de neuronas de entrada 
    modelo = _create_model(x_train.shape[1])
    
    # Veamos los detalles del modelo
    modelo.summary()
    
    # Planificacion con decrecimiento exponencial de la tasa de aprendizaje 
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.1, # tasa de aprendizaje inicial (por defecto en Adam es 0.001)
        decay_steps=50,  
        decay_rate=0.96)

    #  Compilacion
    modelo.compile(loss = 'binary_crossentropy', 
                   optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                              momentum = 0.8),
                   metrics = ['accuracy'])

    # Nombre con el que se guardara el modelo 
    nombreModeloPuntoControl = 'alternative3.keras' 

    # Configuracion del callback
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                filepath=nombreModeloPuntoControl ,
                                monitor='val_accuracy', 
                                mode='max', # maximo del accuracy
                                save_best_only=True, #solamente guardará el mejor y no todos
                                verbose = 1)  # poner a 0 para quitar los mensajes

    # Entrenamiento del modelo
    history= modelo.fit(x_train, 
                     y_train, 
                     epochs = 40, 
                     batch_size = 64,
                     validation_split = 0.2,
                     callbacks=[model_checkpoint_callback])
    
    # Visualizmos el resultado
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

    