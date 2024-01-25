################################################################################
############################### SIGNIFICANCIA ##################################
################################################################################
# SIGNIFICANCE DEFINITION
def significance(df_all):

    df_signal = df_all.query('origin == "signal"')
    df_background = df_all.query('origin == "background"')
    if df_signal.empty:
        # No hay datos de señal
        signal_weight = 0
    else:
        signal_weight = (df_all.loc['signal']["intLumi"]*df_all.loc['signal']["scale1fb"]).sum()

    if df_background.empty:
        # No hay datos de background
        backgrounds_weight = np.nan
    else:
        backgrounds_weight = (df_all.loc['background']["intLumi"]*df_all.loc['background']["scale1fb"]).sum()

    # se calcula la significancia con la fórmula proporcionada
    return np.sqrt(2 * abs( (signal_weight + backgrounds_weight) * np.log(1 + (signal_weight/backgrounds_weight)) - signal_weight))



def barrido_significancia_variable(df_all, variable, derecha = True):
    n_cuts = 100 # numero_iteraciones_cortes
    valores_significancia_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados

    valor_minimo = df_all.loc["signal"][variable].min()
    valor_maximo = df_all.loc["signal"][variable].max()

    df_original = df_all

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):
        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = valor_minimo + i*(valor_maximo-valor_minimo)/n_cuts
        
        if derecha==True:
            df_all = df_original[df_original[variable]>iteration_cut]
        else:
            df_all = df_original[df_original[variable]<iteration_cut]

        # si me quedo sin datos en el signal paro la simulación
        # if df_all.shape[0] == 0:
        #     break
            
        # se calcula la significancia con los nuevos cortes
        significancia_i = significance(df_all)

        # se guarda la significancia y su corte respectivo
        valores_significancia_variable.append(significancia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_significancia_variable


################################################################################
################################ EFICIENCIA ####################################
################################################################################

def efficiency(df, df_cut):
    eficiencia = df_cut.shape[0]/df.shape[0]
    return eficiencia



def barrido_eficiencia_variable(df, variable, derecha = True):
    n_cuts = 100 # numero_iteraciones_cortes
    valores_eficiencias_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados
    n_datos = [] # lista que va a guardar la cantidad de elementos de dataframe
    # n_datos.append(df.shape[0])

    valor_minimo = df[variable].min()
    valor_maximo = df[variable].max()

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):
        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = valor_minimo + i*(valor_maximo-valor_minimo)/n_cuts
        
        if derecha==True:
            df_cut = df[df[variable]>iteration_cut]
        else:
            df_cut = df[df[variable]<iteration_cut]

        # si me quedo sin datos en el signal paro la simulación
        # if df.shape[0] == 0:
        #     break

        # se calcula la significancia con los nuevos cortes
        eficiencia_i = efficiency(df, df_cut)

        # se guarda la significancia y su corte respectivo
        valores_eficiencias_variable.append(eficiencia_i)
        valores_cortes.append(iteration_cut)
        n_datos.append(df_cut.shape[0])
        
    # Se devuelve un DataFrame con dos columnas
    df_eficiencias = pd.DataFrame({
        'n_datos': n_datos,
        'cortes': valores_cortes,
        'eficiencias': valores_eficiencias_variable
    })

    return df_eficiencias
    # return [valores_cortes, valores_eficiencias_variable]

def calc_eficiencias(df_all, variable, derecha = True):
    df_eficiencias = df_all.groupby(["origin", "df_name"]) \
                            .apply(lambda grupo: barrido_eficiencia_variable(grupo, variable, derecha))
    return df_eficiencias


################################################################################
################################ BK REJECTION ##################################
################################################################################


def calc_bk_rejection(df_all, variable, derecha = True):
    df_eficiencias = calc_eficiencias(df_all, variable, derecha)
    df_eficiencias['bk_rejection'] = 1 - df_eficiencias['eficiencias']
    return df_eficiencias



def calc_bk_rejection_all_background(df_all, variable, derecha = True):
    df_background = df_all.query('origin == "background"')
    df_eficiencias = barrido_eficiencia_variable(df_background, variable, derecha)
    df_eficiencias['bk_rejection'] = 1 - df_eficiencias['eficiencias']
    return df_eficiencias


################################################################################
################################### WEIGHT #####################################
################################################################################

def calc_weight(df):
    df_weight = df["intLumi"]*df["scale1fb"]
    return df_weight

# def aplicar_weight(df_all, variable):
#     df_all[variable] = df_all[variable]*calc_weight(df_all)
# PREGUNTAR SI SE APLICA A TO-DO O SOLO ES PARA GRAFICAR

def eficiencia_corte_matriz(matriz):
    eventos_antes = matriz[0][0] + matriz[0][1] + matriz[1][0] + matriz[1][1]
    eventos_despues = matriz[0][0] + matriz[1][0]
    eficiencia = eventos_despues/eventos_antes
    return eficiencia