# -*- coding: utf-8 -*-
"""
This module implement the first deep learning model (grid with hyperparameters)
"""

# Modulos para redes neuronales
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier 
from sklearn.model_selection import GridSearchCV

def modelo1_grid_hyperparams(x_train, y_train):
    # Funcion para crear el modelo, necesaria para KerasClassifier
    def _create_model(neuronasEntrada, neurons):  
        modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(neurons, activation="relu"),
            Dense(1, activation="sigmoid")
            ])
        return modelo



    modelGridSearch = KerasClassifier(model=_create_model,
                                      loss='binary_crossentropy', metrics=['accuracy']) 

    # Definimos los parametros de la busqueda 
    # 3x3x2x3 = 54 combinaciones de paramteros
    neuronas = [32, 64, 96]
    opt = ['sgd', 'rmsprop', 'adam'] #son los 3 más utilizados
    epocas= [20,40]
    lotes = [64, 128, 256]

    #Creando el modelo con la rejilla
    param_grid = dict(model__neuronasEntrada = [x_train.shape[1]],
                      model__neurons=neuronas,
                      optimizer=opt,
                      epochs=epocas, 
                      batch_size=lotes,
                      verbose = [0]) 

    # Definiendo cross-validation
    grid = GridSearchCV(estimator=modelGridSearch, #modelo a ejecutar
                        param_grid=param_grid, 
                        cv = 3, 
                        verbose=3) 
    grid_result = grid.fit(x_train, y_train)

    # Resumen resultados
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) con: %r" % (mean, stdev, param))
    print("Mejor: %f utilizando %s" % (grid_result.best_score_, grid_result.best_params_)) #se queda con el que tiene mejor accuracy y cuales son su parametros

    # Guardamos el mejor modelo encontrado
    bestmodel = grid_result.best_estimator_.model_
    bestmodel.save('modeloPractica.keras')
# metricsGridSearchCV = bestmodel.evaluate(x_test, y_test)    
