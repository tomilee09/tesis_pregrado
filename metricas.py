import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.pylab import rcParams
from xgboost import plot_tree
import numpy as np
import pandas as pd

# probando la curva de roc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# grafico las desiciones que tomo el arbol
def grafico_decisiones(modelo):
    rcParams['figure.figsize'] = 200,500
    plot_tree(modelo)
    plt.show()



# ver presición del modelo
def precision(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy * 100:.2f}%')



# entrega la matríz del gráfico de confusión, no grafíca
#                 Predicción Positivo    Predicción Negativo
# Actual Positivo        TP                    FN
# Actual Negativo        FP                    TN

# NOTESE QUE EL FORMATO EN QUE LO ENTREGA SKLEARN POR ALGÚN MOTIVO ES:
#                 Predicción Positivo    Predicción Negativo
# Actual Positivo        TN                    FP
# Actual Negativo        FN                    TP
def confusion(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    m = confusion_matrix(y_test, y_pred)
    matriz = [[m[1][1], m[1][0]], [m[0][1], m[0][0]]]
    return matriz
    # display_matriz = ConfusionMatrixDisplay(confusion_matrix=matriz)
    # display_matriz.plot()
    # plt.show()


# gráfico de confusión de lo predicho respecto a lo que de verdad da
def plot_matriz(matrix, porcentaje, normalizado):
    clases = ["VBF", "ggF"]

    # normalizo los datos dependiendo si son vbf o ggf
    if normalizado == True:
        n_datos_vbf = matrix[0][0]+matrix[0][1]
        n_datos_ggf = matrix[1][0]+matrix[1][1]
        matriz_confusion = [[matrix[0][0]/n_datos_vbf,matrix[0][1]/n_datos_vbf],
                            [matrix[1][0]/n_datos_ggf,matrix[1][1]/n_datos_ggf]]
    else: 
        matriz_confusion = matrix


    # Crear una figura y un subplot
    plt.figure(figsize=(6, 6)) # para que no se vea tan grande
    plt.matshow(matriz_confusion, cmap="Blues", fignum=1)

    # Agregar etiquetas
    for i in range(2):
        for j in range(2):
            plt.text(j, i, np.format_float_scientific(matriz_confusion[i][j], precision=3), ha="center", va="center", fontsize=14, color="r")

    # Configuración de ejes y etiquetas
    plt.xticks([0, 1], clases)
    plt.yticks([0, 1], clases)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Valor Real", fontsize=12)
    plt.title("Matriz de Confusión", fontsize=14)

    # Mostrar el gráfico
    plt.show()



# es la matriz de confusión pero hecho en gráfico de barras en vez de un cuadrado
def plot_confusion_barras(matrix):

    # normalizo los datos de vbf y ggf
    n_datos_vbf = matrix[0][0]+matrix[0][1]
    n_datos_ggf = matrix[1][0]+matrix[1][1]
    datos = [matrix[0][0]/n_datos_vbf,matrix[0][1]/n_datos_vbf,
            matrix[1][0]/n_datos_ggf,matrix[1][1]/n_datos_ggf]

    # creo un dataframe para realizar el gráfico más facilmente con seaborn
    matriz_graficar = {'normalized value':datos,
                       'true origin':["VBF", "VBF", "ggF", "ggF"],
                       'classified as':["VBF", "ggf", "VBF", "ggf"],}
    df_matriz = pd.DataFrame(matriz_graficar)

    # realizo el gráfico
    plt.figure(figsize=(6, 6)) # para que no se vea tan grande
    ax = sns.barplot(data = df_matriz,
                x='classified as', 
                y ='normalized value',
                hue = "true origin",
                )
    
    # Agregar los valores encima de las barras
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')
    
    ax.legend(title="true origin", loc="upper center")

    plt.show()



# busca las correlaciones más importantes con la variable seleccionada
def correlaciones(df, variable):
    correlation_matrix = df.corr()
    target_correlations = correlation_matrix[variable].sort_values(ascending=False)
    return target_correlations



# grafica la correlación en los parámetros de un dataframe
def plot_correlacion(df):
    correlacion = df.corr()
    # correlacion.style.background_gradient(cmap='coolwarm')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                square=True, 
                vmin=-1, 
                vmax=1,
                # annot_kws={"size": 12}
                )
    # plt.matshow(correlacion)
    plt.show()



# hacer prueba de Kolmogorov-Smirnov
# aparentemente sirve para ver si se sigue un tipo de distribución
# es la resta entre las 2 distribuciones, así que si es alta significa que son muy diferentes
# (en específico es la resta entre la acumulacion de las 2 distribuciones (las cumulative))
from scipy.stats import kstest
def prueba_ks(df, variable, tipo_distribucion_esperada):
    datos = df[variable]

    statistic, p_valor = kstest(datos, tipo_distribucion_esperada)

    # Establece un nivel de significancia
    alfa = 0.05

    # Compara el p-valor con el nivel de significancia
    if p_valor < alfa:
        print(f"Se rechaza la hipótesis nula, p vale: {p_valor}. Los datos no siguen una distribución {tipo_distribucion_esperada}.")
    else:
        print(f"No se rechaza la hipótesis nula, p vale: {p_valor}. Los datos siguen una distribución {tipo_distribucion_esperada}.")




# curva de ROC
def plot_ROC(modelo, X_test, y_test):
    y_predict = modelo.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot(c = "red", label="ml") 
    plt.plot([0, 1], c="blue", label="random")
    plt.legend(loc="right")
    plt.show() 