from formulas import *
from cortes import *
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
################################# GRAFICAR #####################################
################################################################################



def graficar(df_all, variable, derecha = True, graficar_significancia = True, graficar_eficiencia = True, aplicar_weights = True):

    # configuraciones para el gráfico
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 24 # estaba en 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')


    ################# ELIJO LA FORMA DEL GRAFICO DEPENDIENDO LO QUE QUEREMOS GRAFICAR ################


    if ((graficar_eficiencia == True) and (graficar_significancia == True)):
        fig, axes = plt.subplots(3,1, figsize=(10,12), sharex=True, sharey=False,  gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]})
        #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #fig.tight_layout(fig)
        
        ###################################### TO DO : ################################
        ###################### CORREGIR LOS BORDES DEL GRAFICO #######################
        
        eje_histograma = axes[0]
        eje_significancia = axes[1]
        eje_eficiencia = axes[2]

    elif ((graficar_eficiencia == False) and (graficar_significancia == False)):
        fig, axes = plt.subplots(1,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes

    elif ((graficar_eficiencia == True) and (graficar_significancia == False)):
        fig, axes = plt.subplots(2,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes[0]
        eje_eficiencia = axes[1]
    
    else: # ((graficar_eficiencia == False) and (graficar_significancia == True))
        fig, axes = plt.subplots(2,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes[0]
        eje_significancia = axes[1]


    ################## MODIFICACION DATOS PARA GRAFICAR ##########################


    # pocentajes 
    porcentaje_bajo = 0.02
    porcentaje_alto = 0.98

    # elimino los valores extremos de df_all
    low_data = df_all.loc["signal"][variable].quantile(porcentaje_bajo)
    high_data  = df_all.loc["signal"][variable].quantile(porcentaje_alto) 
    df_all = df_all[(df_all[variable]>low_data) & (df_all[variable]<high_data)]

    best_cut_eficiencia = find_best_cut(df_all, variable, "eficiencia", derecha)
    best_cut_significancia = find_best_cut(df_all, variable, "significancia", derecha)


    ################## GRAFICO DE SIGNIFICANCIA ##########################


    if graficar_significancia == True:
        # calculo la significancia de la variable introducida
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable, derecha)
        #Scatter de la significancia.
        scatter_significancia = sns.scatterplot(ax = eje_significancia, x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
        scatter_significancia.set_xlabel(variable, fontdict={'size':12})
        scatter_significancia.set_ylabel('Significance', fontdict={'size':12})
        scatter_significancia.axvline(x = best_cut_significancia, color = 'red', label = 'corte significancia')
 

    # datos previos de los histogramas
    # color_palette = sns.color_palette("hls", len(backgrounds))
    # my_binwidth = (df_all.loc["signal"][variable].max() - df_all.loc["signal"][variable].min())/100.
    n_bins = 50

    ################## HISTOGRAMA DE LOS DATOS ##########################
    # if aplicar_weights == True:
    #     weight_signal = calc_weight(signal)
    # else:
    #     weight_signal = np.ones(signal.shape[0])


    histoplot = sns.histplot(ax=eje_histograma, 
                             data=df_all, 
                             x=variable, 
                             hue=df_all.index.get_level_values('df_name'),
                             legend=True,
                             alpha=0.05,  
                             stat='density', 
                             common_norm=False, 
                             binrange=(df_all.loc["signal"][variable].min(), df_all.loc["signal"][variable].max()), 
                             binwidth = (df_all.loc["signal"][variable].max() - df_all.loc["signal"][variable].min())/n_bins,  
                             weights=calc_weight(df_all), 
                             element="step", 
                            #  fill=False,
                             )
    

    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(str(variable), fontdict={'size':12})
    histoplot.set_ylabel('Normalised Events for ' + str(variable) , fontdict={'size':12})
    histoplot.ticklabel_format(style='plain', axis='y')
    histoplot.axvline(x = best_cut_significancia, color = 'red', label = 'corte significancia')
    histoplot.axvline(x = best_cut_eficiencia, color = 'blue', label = 'corte significancia')


    ################## GRAFICO DE EFICIENCIA ##########################
    # color_palette = sns.color_palette("hls", len(backgrounds))
    if graficar_eficiencia == True:

        ######################### EFICIENCIA ##########################
        
        eficiencias_signal = calc_eficiencias(df_all, variable, derecha).query('origin=="signal"')
        scatter_eficiencia = sns.scatterplot(ax = eje_eficiencia, 
                                             data=eficiencias_signal, 
                                             x = "cortes", 
                                             y = "eficiencias", 
                                             color = 'black', 
                                            #  hue=df_eficiencias.index.get_level_values('df_name'),
                                             #legend=True,
                                             marker=(8,2,0), 
                                             s=75)


        # modificaciones graficos eficiencias
        scatter_eficiencia.set_xlabel(variable, fontdict={'size':12})
        scatter_eficiencia.set_ylabel('Efficiency', fontdict={'size':12})
        
        ######################### REJECTION ##########################

        # calculo y grafico el background rejection de signal
        bk_rejection_background = calc_bk_rejection(df_all, variable, derecha).query('origin == "background"')
        sns.scatterplot(data=bk_rejection_background, 
                        x="cortes", 
                        y="bk_rejection", 
                        # color = 'black', 
                        hue=bk_rejection_background.index.get_level_values('df_name'),
                        #label = "bk rejection signal",
                        #legend=True, 
                        marker=(8,2,0), 
                        s=30
                        )

        # calculo el background rejection de todos los backgrounds unidos
        bk_rejection_all_background = calc_bk_rejection_all_background(df_all, variable, derecha)
        sns.scatterplot(data=bk_rejection_all_background, 
                        x="cortes", 
                        y="bk_rejection", 
                        color = 'black', 
                        # hue=bk_rejection_background.index.get_level_values('df_name'),
                        #label = "bk rejection signal",
                        #legend=True, 
                        marker=(8,2,0), 
                        s=30
                        )
        
        # grafico de la linea de corte
        scatter_eficiencia.axvline(x = best_cut_eficiencia, color = 'blue', label = 'corte eficiencia')
        
        ####################### BARRAS DE ERROR ##########################
        #Calculo la desviación estándar de la lista eficiencia y se agregan al gráfico de eficiencia.
        # std = np.std(eficiencia_variable)
        plt.errorbar(x = eficiencias_signal["cortes"], 
                     y = eficiencias_signal["eficiencias"], 
                     yerr = np.sqrt(1/eficiencias_signal["n_datos"]), # multiplicar por desviacion estandar?, tendría que guardarla en la funcion de las eficiencias
                     fmt='none', 
                     linestyle='none')
        #plt.xlim(0,1000)
    
    
    
    #plt.savefig('complete_Graph1_MET.eps', format = 'eps')
    #plt.savefig('complete_Graph1_MET.pdf', format = 'pdf')
    #plt.legend()
    plt.show()

    ############## VIOLIN PLOT ####################
    # sns.violinplot(data=df_all, x=variable, y=df_all.index.get_level_values('df_name'))
    # plt.show()

    ############## PIE PLOT ####################
    # df_all.groupby(level='origin').size().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    # plt.show()