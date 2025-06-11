"""
This module validates the model saved in the module 'model3_exponential_decay.py'
"""
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def model3_train_validation(modelo3,x_train,y_train):
    # Prediccion
    predicciones = modelo3.predict(x_train) 
    modelo3.evaluate(x_train, y_train)
    print(predicciones[0:10,])
    # Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
    predic_train = 1*(predicciones>0.5)
    print(predic_train[0:10,])
    # Datos reales
    print(y_train[0:10,])
    
    # Matriz de contingencia o matriz de confusion
    mc_train = confusion_matrix(y_train, predic_train)
    print(mc_train)
    
    # Graficamos la matriz
    class_names = ['NoChurn', 'Churn']
    disp = ConfusionMatrixDisplay(confusion_matrix = mc_train, display_labels = class_names)
    disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


def model3_test_validation(modelo3,x_test,y_test):
    predicciones2 = modelo3.predict(x_test)
    modelo3.evaluate(x_test, y_test)
    print(predicciones2[0:10,])
    # Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
    predic_test = 1*(predicciones2>0.5)
    print(predic_test[0:10,])
    # Datos reales
    print(y_test[0:10,])
    # Matriz de contingencia o matriz de confusion
    mc_test = confusion_matrix(y_test, predic_test)
    print(mc_test)
    
    # Graficamos la matriz
    class_names = ['NoChurn', 'Churn']
    disp = ConfusionMatrixDisplay(confusion_matrix = mc_test, display_labels = class_names)
    disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
