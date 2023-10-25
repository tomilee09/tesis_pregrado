import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
############################### DO CUTS ##################################
##########################################################################

#Se le da el corte
# se realizan los cortes superiores e inferiores a una lista de dataframes
def do_cuts(df_all, cuts, scale):

    # se aplican todos los cortes
    for variable in cuts:
        # se grafica el numero de eventos
        numero_eventos_antes = df_all.shape[0]
        print(f'Numero eventos antes: {numero_eventos_antes}')
        
        #Definimos si el corte es un booleano. Se ocupa cuando se quieren cortar cosas del estilo Triggers
        if type(cuts[variable]) == type(True):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]

            
        #Definimos si el corte es una lista, esto se ocupa para cuando se quieren cortar máximos y mínimos.   
        elif type(cuts[variable]) == type([]):
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]
            
            print(f'Corte: {variable} entre {cuts[variable]}')
            df_all = df_all[df_all[variable] < corte_mayor]
            df_all = df_all[df_all[variable] > corte_menor]

        
        #Definimos si el corte es un número entero. Se ocupa para cuando queremos separar los datos que tienen que ser un valor específico como un veto.
        elif type(cuts[variable]) == type(0):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]


        # elif type(cuts[variable]) == type(''):
        #     print(f'Corte: {variable} == {cuts[variable]}')
        #     df_all = df_all[df_all[variable] == cuts[variable]]
            
        else:
            print("ADVERTENCIA: NO TOMA LA VARIABLE DEL CORTE")

        numero_eventos_despues = df_all.shape[0]
        print(f'Numero eventos despues: {numero_eventos_despues} \n')
    
    # elimino los pesos negativos
    # weights = df_all["intLumi"]*df_all["scale1fb"]
    df_all = df_all[df_all["intLumi"]*df_all["scale1fb"] >= 0]
             
    return df_all


################################################################################
############################### FIND BEST CUT ##################################
################################################################################


def find_best_cut(df_all, variable, method, derecha = True):
    if method == "significancia":
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable, derecha)
        index_max_significance = significancia_variable.index(max(significancia_variable))
        maximo_corte = cortes[index_max_significance]
        return maximo_corte
        
    if method == "eficiencia":
        eficiencias_signal = calc_eficiencias(df_all, variable, derecha).query('origin=="signal"')["eficiencias"]

        bk_rejection = calc_bk_rejection_all_background(df_all, variable, derecha)
        bk_rejection_bk = bk_rejection["bk_rejection"]

        diferencia = abs(bk_rejection_bk.reset_index(drop=True)-eficiencias_signal.reset_index(drop=True))
        indice_minima_diferencia = diferencia.idxmin()
        corte_interseccion = bk_rejection["cortes"][indice_minima_diferencia]
        
        return corte_interseccion



################################################################################
################################# TESTING CUT ##################################
################################################################################

# se realizan los cortes superiores e inferiores a una lista de dataframes
def test_cuts(df_all, cuts, scale):

    df_original = df_all

    significancias = []
    eficiencias = []
    cortes = []
    variables = []
    n_datos = []
    n_datos_signal = []
    n_datos_background = []
    peso_signal = []

    significancias.append(significance(df_all))
    eficiencias.append(efficiency(df_all, df_all))
    cortes.append(0)
    variables.append("")
    n_datos.append(df_all.shape[0])
    n_datos_signal.append(df_all.query('origin=="signal"').shape[0])
    n_datos_background.append(df_all.query('origin=="background"').shape[0])
    peso_signal.append((df_all.query('origin=="signal"')["intLumi"]*df_all.query('origin=="signal"')["scale1fb"]).sum())

    print(f'datos antes corte: {n_datos[-1]}')
    print(f'datos background antes corte: {n_datos_background[-1]}')
    print(f'datos signal antes corte: {n_datos_signal[-1]}')
    print(f'peso signal antes corte: {peso_signal[-1]} \n')

    # si se queda vacío puedo usar este para devolver las etiquetas
    df_recover_bk = df_all.query('origin=="background"').iloc[:2]*0 
    df_recover_sg = df_all.query('origin=="signal"').iloc[:2]*0

    # se aplican todos los cortes
    for variable in cuts:

        #Definimos si el corte es un booleano. Se ocupa cuando se quieren cortar cosas del estilo Triggers
        if type(cuts[variable]) == type(True):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]

            
        #Definimos si el corte es una lista, esto se ocupa para cuando se quieren cortar máximos y mínimos.   
        elif type(cuts[variable]) == type([]):
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]
            
            print(f'Corte: {variable} entre {cuts[variable]}')
            df_all = df_all[df_all[variable] < corte_mayor]
            df_all = df_all[df_all[variable] > corte_menor]

        
        #Definimos si el corte es un número entero. Se ocupa para cuando queremos separar los datos que tienen que ser un valor específico como un veto.
        elif type(cuts[variable]) == type(0):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]


        elif type(cuts[variable]) == type(''):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]
            
        else:
            print("ADVERTENCIA: NO TOMA LA VARIABLE DEL CORTE")
             
    
        #
        # si alguno queda vacío vuelvo a poner elementos con las etiquetas, pero de valor 0, y después paro el código
        #

        df_signal = df_all.query('origin == "signal"')
        df_background = df_all.query('origin == "background"')

        if df_signal.empty:
            # No hay datos de señal
            print("Quedamos sin datos de señal")
            df_all = pd.concat([df_all, df_recover_sg])

        if df_background.empty:
            # No hay datos de background
            print("Quedamos sin datos de background")
            df_all = pd.concat([df_all, df_recover_bk])

        # guardamos los datos
        significancias.append(significance(df_all))
        eficiencias.append(efficiency(df_original, df_all))
        cortes.append(df_all[variable])
        variables.append(variable)
        n_datos.append(df_all.shape[0])
        n_datos_signal.append(df_all.query('origin=="signal"').shape[0])
        n_datos_background.append(df_all.query('origin=="background"').shape[0])
        peso_signal.append((df_all.query('origin=="signal"')["intLumi"]*df_all.query('origin=="signal"')["scale1fb"]).sum())

        # se para el código
        if df_signal.empty:
            break

        if df_background.empty:
            break

        # print(f'datos despues corte: {n_datos[-1]}')
        # print(f'datos background despues corte: {n_datos_background[-1]}')
        # print(f'datos signal despues corte: {n_datos_signal[-1]}')
        # print(f'peso signal despues corte: {peso_signal[-1]} \n')

        

    
    df_data = pd.DataFrame({
        #'cortes': cortes,
        'variables': variables,
        # 'n_datos': n_datos,
        'n_datos_background': n_datos_background,
        'n_datos_signal': n_datos_signal,
        'eficiencias': np.round(eficiencias, 3),
        'significancias': np.round(significancias, 3),
        'weights': peso_signal
    })
    
    print(df_data)

    ####### STACKED ############
    # sns.barplot(data = df_data, x = "variables", y = "n_datos")
    # df_data.plot(x='variables', kind='bar', stacked=True,
    #     title='Stacked Bar Graph by dataframe')
    # axes = df_data.plot.bar(rot=0, subplots=True)
    # axes[1].legend(loc=2)  
    
    ax1 = plt.subplot(1,1,1)
    w = 0.3
    
    #plt.xticks(), will label the bars on x axis with the respective country names.
    x = np.arange(df_data['variables'].shape[0])
    plt.xticks(x + w /2, df_data['variables'], rotation='vertical')

    cmap_rojo =plt.get_cmap("Reds")
    cmap_azul =plt.get_cmap("Blues")

    valores_normalizados_signal = (df_data['n_datos_signal']) / (df_data['n_datos_signal'].max())
    valores_normalizados_background = (df_data['n_datos_background']) / (df_data['n_datos_background'].max())

    cmap_azul_signal = cmap_azul(valores_normalizados_signal)
    cmap_rojo_background = cmap_rojo(valores_normalizados_background)

    # creo las barras del background
    background =ax1.bar(x, df_data['n_datos_background'], width=w, color=cmap_rojo_background, align='center', yerr=1/np.sqrt(df_data['n_datos_background']))
    
    # pongo los label de cada barra
    ax1.bar_label(background, labels=df_data['significancias'], label_type='edge', padding=40)
    ax1.bar_label(background, labels=df_data['eficiencias'], label_type='edge', padding=20)
    
    # pongo los porcentajes al centro
    porcentaje_signal = np.round((df_data['n_datos_signal']/(df_data['n_datos_signal'] + df_data['n_datos_background']) )*100, 1)
    porcentaje_background = np.round((df_data['n_datos_background']/(df_data['n_datos_signal'] + df_data['n_datos_background']) )*100, 1)
    ax1.bar_label(background, labels=porcentaje_background, label_type='center')

    # creo las barras de la señal
    ax2 = ax1.twinx()
    signal =ax2.bar(x + w, df_data['n_datos_signal'], width=w, color=cmap_azul_signal, align='center', yerr=1/np.sqrt(df_data['n_datos_signal']))
    
    # pongo el porcentaje del centro de signal
    ax2.bar_label(signal, labels=porcentaje_signal, label_type='center')

    # pongo los ejes logaritmicos
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    #Set the Y axis label as GDP.
    plt.ylabel('N datos')
    #To set the legend on the plot we have used plt.legend()
    plt.legend([background, signal],['background', 'signal'])
    #To show the plot finally we have used plt.show().
    plt.show()