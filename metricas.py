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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
import matplotlib

# grafico las desiciones que tomo el arbol
def grafico_decisiones(modelo):
    rcParams['figure.figsize'] = 200,500
    plot_tree(modelo)
    plt.show()



# ver presición del modelo
def precision(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_pred = y_pred>0.5
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
def confusion(y_test, y_pred):
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
def plot_confusion_barras(matrix ,normalizado=True):
    
    # Configurar el color del fondo y de las barras
    fondo_color = "#05192d"
    colores_hue = {"ggF": "#FF5733", "VBF": "#03ef62"}
    blanco = "#ffffff"

    # normalizo los datos de vbf y ggf
    n_datos_vbf = matrix[0][0]+matrix[0][1]
    n_datos_ggf = matrix[1][0]+matrix[1][1]
    
    if normalizado == True:
        datos = [matrix[1][1]/n_datos_ggf,matrix[1][0]/n_datos_ggf,
                 matrix[0][1]/n_datos_vbf,matrix[0][0]/n_datos_vbf]

    else:
        datos = [matrix[1][1],matrix[1][0],
                 matrix[0][1],matrix[0][0]]

    # creo un dataframe para realizar el gráfico más facilmente con seaborn
    matriz_graficar = {'normalized value':datos, # cambié el orden para estar de acuerdo con los otros gráficos
                       'true origin':["ggF", "ggF", "VBF", "VBF"],
                       'classified as':["ggF", "VBF", "ggF", "VBF"],}
    df_matriz = pd.DataFrame(matriz_graficar)

    # realizo el gráfico
    plt.figure(figsize=(6, 6)) # para que no se vea tan grande
    ax = sns.barplot(data = df_matriz,
                x='classified as', 
                y ='normalized value',
                hue = "true origin",
                palette=colores_hue
                )
    
    if normalizado == False:
        ax.set_ylabel("number of events")
    
    # color del fondo
    ax.set_facecolor(fondo_color)
    plt.gcf().set_facecolor(fondo_color)
    
    # Cambiar el color del borde del gráfico
    ax.spines['top'].set_edgecolor(fondo_color)
    ax.spines['right'].set_edgecolor(fondo_color)
    ax.spines['bottom'].set_edgecolor(blanco)
    ax.spines['left'].set_edgecolor(blanco)
        
    # Cambiar el color de las letras y números
    color_letras_numeros = "#ffffff"
    ax.tick_params(axis='both', colors=color_letras_numeros)
    ax.xaxis.label.set_color(color_letras_numeros)
    ax.yaxis.label.set_color(color_letras_numeros)
    ax.title.set_color(color_letras_numeros)
    
    # Ajustar el tamaño de los números en los ejes
    tamanio_numeros_ejes = 12
    ax.tick_params(axis='both', labelsize=tamanio_numeros_ejes)
    
    # Ajustar el tamaño de las etiquetas de los ejes y el título
    tamanio_etiquetas_ejes = 12
    ax.xaxis.label.set_size(tamanio_etiquetas_ejes)
    ax.yaxis.label.set_size(tamanio_etiquetas_ejes)
    ax.title.set_size(tamanio_etiquetas_ejes)
    
    # Agregar los valores encima de las barras
    if normalizado == True:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='white', xytext=(0, 10),
                        textcoords='offset points')
            
    else:
        for p in ax.patches:
            value = p.get_height()
            ax.annotate('{:,.0f}'.format(value), 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='white', xytext=(0, 10),
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
    





# esto es un decorador para hacer lindo el gráfico
def dar_estilo_plt(funcion):
    def wrapper(*args, **kwargs):
        # {"ggF": "#FF5733", "VBF": "#03ef62"}
        fondo_color = "#05192d"
        blanco = "#ffffff"
        colores_hue = {"ggF": "#FF5733", "VBF": "#03ef62"}

        fig, ax = funcion(*args, **kwargs)

        # Personalizar el estilo del gráfico
        fig.set_facecolor(fondo_color)
        ax.set_facecolor(fondo_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(blanco)
        ax.spines['left'].set_color(blanco)
        ax.tick_params(axis='x', colors=blanco)
        ax.tick_params(axis='y', colors=blanco)

        plt.show()

    return wrapper

@dar_estilo_plt
def plot_cantidad_ggf_vbf(n_ggF, n_VBF):
    fig, ax = plt.subplots(figsize=[5, 5])
    plt.bar("ggF", n_ggF, label='Value 1', color='#FF5733', edgecolor='none')
    plt.bar("VBF", n_VBF, label='Value 2', color='#03ef62', edgecolor='none')

    plt.ylabel('number of events', color='white')
    
    # Devolver fig, ax para que pueda ser personalizado por el decorador
    return fig, ax



@dar_estilo_plt
def plot_number_events_hist(modelo, X_test):
    y_pred = modelo.predict(X_test)
    fig, ax = plt.subplots(figsize=[5, 5])
    plt.hist(y_pred, bins = 20, edgecolor='white') # antes usaba sns
    plt.ylabel('number of events', color='white')
    plt.xlabel('probability of VBF', color='white')
    
    return fig, ax




from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Decorador para personalizar el estilo del gráfico
def dar_estilo_sns(func):
    def wrapper(*args, **kwargs):
        # Configuración del estilo del gráfico
        fondo_color = "#05192d"
        blanco = "#ffffff"
        colores_hue = {"ggF": "#FF5733", "VBF": "#03ef62"}

        # Llamar a la función decorada con los argumentos y argumentos de palabra clave
        fig, ax = func(*args, **kwargs)

        # Personalizar el estilo del gráfico
        ax.set_facecolor(fondo_color)
        plt.gcf().set_facecolor(fondo_color)
        ax.spines['top'].set_edgecolor(fondo_color)
        ax.spines['right'].set_edgecolor(fondo_color)
        ax.spines['bottom'].set_edgecolor(blanco)
        ax.spines['left'].set_edgecolor(blanco)

        color_letras_numeros = "#ffffff"
        ax.tick_params(axis='both', colors=color_letras_numeros)
        ax.xaxis.label.set_color(color_letras_numeros)
        ax.yaxis.label.set_color(color_letras_numeros)
        ax.title.set_color(color_letras_numeros)

        tamanio_numeros_ejes = 12
        ax.tick_params(axis='both', labelsize=tamanio_numeros_ejes)

        tamanio_etiquetas_ejes = 12
        ax.xaxis.label.set_size(tamanio_etiquetas_ejes)
        ax.yaxis.label.set_size(tamanio_etiquetas_ejes)
        ax.title.set_size(tamanio_etiquetas_ejes)

        plt.show()

    return wrapper

# Plot básico
@dar_estilo_sns
def plot_pca_2D(model, df_X, df_y):
    pos = [0.0, 1.0]
    ggF = "#FF5733"
    VBF = "#03ef62"
    colors = [ggF, VBF]
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    try:
        register_cmap("cmap_presentacion", cmap)
    except:
        pass

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_X)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    df_plot = pd.DataFrame()
    df_plot["x"] = x_pca[:, 0]
    df_plot["y"] = x_pca[:, 1]
    df_plot["color"] = df_y.to_numpy()

    custom_palette = sns.color_palette([ggF, VBF])
    cmap_custom = LinearSegmentedColormap.from_list('custom_palette', [ggF, VBF])

    plt.figure(figsize=[5, 5])

    ax = sns.scatterplot(data=df_plot,
                         x="x",
                         y="y",
                         hue="color",
                         palette=custom_palette,
                        #  palette=sns.color_palette('cmap_presentacion', as_cmap = True), 
                         alpha=0.1,
                         linewidth=0)

    return plt.gcf(), ax