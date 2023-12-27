# Clasificación entre VBF y ggF

Código realizado para mejorar la clasificación entre eventos ggF y VBF en los decaimientos de bosones de Higgs para una mejor detección de fotones oscuros.

Se utilizan los filtros utilizados previamente, filtro creado mediante la clasificación con Machine Learning (XGBOOST) y filtros creados mediante Deep Learning (Tensorflow).

Los archivos son los siguientes:

## .ipynb
- analisis_cortes.ipynb: código que aplica los filtros sin utilizar machine learning.
- analisis_xgboost.ipynb: código que realiza un análisis de los datos, luego realiza un entrenamiento de un modelo de machine learning utilizando el módulo XGBOOST. Finalmente se realiza un análisis de la clasificación, viendo el porcentaje de logro, y si el modelo tiene overtraining.
- analisis_tensorflow.ipynb: código que realiza un entrenamiento de los datos utilizando Deep Learning con el módulo Tensorflow.
- analisis_recurrente.ipynb y analisis_attention.ipynb: código experimental para comprobar la efectividad de utilizar redes neuronales recurrentes y redes neuronales con atención, de momento tienen la misma efectividad que su no uso.

## .py
- lectura.py: funciones utilizadas para la lectura de los archivos .root, ordenandolos en variables tipo DataFrame del módulo Pandas.
- machine_learning.py: funciones para el entrenamiento y la búsqueda del mejor modelo de machine learning.
- metricas.py: funciones que entregan las métricas de los modelos entrenados, su output puede ser valores numéricos o gráficos.
- cortes.py: aplica los filtros que se le entrega, además de encontrar el mejor corte para aumentar la significancia.
- formulas.py y graficar.py: actualmente no usados, fueron utilizados para encontrar el mejor filtro para cada variable del modelo.

## .yaml
- parametros_cortes.yaml: archivo que contiene los datos previos del modelo, este archivo contiene además todos los filtros oficiales del modelo utilizados hasta el momento.
- parametros_ml.yaml: archivo que contiene los datos previos del modelo, este archivo contiene algunos de los cortes oficiales, pero no se realizan otros cortes para que el modelo aprenda cuales son los mejores cortes para realizar la clasificación.

## .pdf
- advances_in_classification.pdf: presentación con los avances realizados hasta el 20 de noviembre del 2023.
