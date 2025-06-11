"""
Modelo de red neuronal para predecir si un cliente de una empresa de telecomunicación
iraniana será churn (cancelado) o no.
"""

from tensorflow import keras
import processing_data as pr
import alternative1 as model1
import alternative2 as model2
import alternative3 as model3
import alternative3_validation as valid_model3

# Paquete para el dataset
from ucimlrepo import fetch_ucirepo

# Importando los datos
# fetch dataset 
iranian_churn = fetch_ucirepo(id=563) 
  
# data (as pandas dataframes) 
X = iranian_churn.data.features 
y = iranian_churn.data.targets

# Ejecutando el modulo processing_data
    # Verificando datos faltantes
    # Aplicando las transformaciones necesarias en los datos numericos y categoricos
    # Ajustando la proporción de los datos por su clase
x_train,x_test,y_train,y_test = pr.processing_data(X,y)

# Ejecutando el primer modelo tras definir los hiperparametros en una rejilla
    # Hiperparametros de la rejilla:
    # neuronas = [32, 64, 96]
    # opt = ['sgd', 'rmsprop', 'adam'] #son los 3 más utilizados
    # epocas= [20,40]
    # lotes = [64, 128, 256]
# Objetivo: probar con distintos hiperparametros, cual mejor se adecua a estos
# datos.
model1.modelo1_grid_hyperparams(x_train, y_train)

# Mejor modelo:
# Mejor: 0.942381 utilizando {'batch_size': 64, 'epochs': 40, 
# 'model__neuronasEntrada': 30, 'model__neurons': 96, 'optimizer': 'adam', 
# 'verbose': 0}

# También fue analizado que el stdev del mejor modelo es bajo: (0.003883)

# Como segundo modelo, se utilizan los hiperparametros del modelo anterior
# que tuvieron mejor accuracy, para poder analisar sus métricas de evaluación
model2.modelo2_best_grid(x_train, y_train)
# accuracy: 0.956845 - loss: 0.125376 - val_accuracy: 0.944048 - val_loss: 0.156907

# Tras analizar los gráficos y la matrix de confusión, se nota que el modelo 
# todavía podría aprender más, ya que el gráfico tiene tendencia de crecimiento,
# no se ha estancado. Los gráficos de entrenamiento y de validación están un poco
# alejados, pero no suficientemente para que sea considerado overfitting.
# Además, se nota que el fallo en la predicción es 
# superior en los clientes que no son churn, comparado con los churn.

# Como el gráfico del modelo anterior tiene tendencia de crecimiento, se aplica 
# en el modelo siguiente (modelo 3), una tasa de aprendizaje adaptativa y callback:
model3.modelo3_exponential_decay(x_train, y_train)

# Mejor accuracy en la época 39: 
# accuracy: 0.965774 - loss: 0.105120 - val_accuracy: 0.965476 - val_loss: 0.127134

# Abriendo el modelo3 salvo (época 39)
modelo3 = keras.models.load_model('alternative3.keras')
modelo3.summary()

# Validando el modelo 3 (época 39) en los datos de entrenamiento
valid_model3.model3_train_validation(modelo3,x_train,y_train)

# Accuracy en los datos de entrenamiento:
# accuracy: 0.9692 - loss: 0.1048

# El modelo 3 tiene mejor resultado comparado con los demás modelos, pero
# no se nota una diferencia significativa en el accuracy de entrenamiento de 
# cada uno de los modelos (variación del accuracy de los datos de validación 
# del entrenamiento: 0.944048 - 0.965476).

# Una vez elegido el modelo, se lo aplica en los datos de test

# Validando el modelo 3 en los datos de test
valid_model3.model3_test_validation(modelo3,x_test,y_test)
# Accuracy en los datos de test:
# accuracy: 0.9614 - loss: 0.1139 

# Además de los comentarios anteriores, para validar el modelo elegido, se verifica
# si la métrica de evaluación se comporta de manera similar entre los datos de 
# entrenamiento y de validación. El modelo 3 continua siendo apto para ser utilizado,
# visto que tiene un accuracy del 96.14% en los datos de test.