# import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm # sirve para ver la linea de carga al cargar los archivos
# import sys
import yaml

# Leer archivos de data_.yaml
def read_data_yaml(data_yaml_file):
    with open(data_yaml_file) as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return data_yaml


#Leer archivos root
def read_root_file(path, filename, tree_name):
    file = uproot.open(path + filename)
    tree = file[tree_name]
    return tree


#Se define la función con la que se escalan las variables para transformar unidades.
def scale_df(df, scale):
    for variable in scale:
        df[variable] = df[variable]*scale[variable]
    return df


# crea una lista con todos los datasets introducidos en datasets
def read_datasets(vbf_data, ggf_data, variables, scale, path):
    df_all = pd.DataFrame()
    df_all_signal = pd.DataFrame()
    df_all_background = pd.DataFrame()

    # se leen los df's introducidos en datasets
    for data in tqdm(vbf_data): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        # añado llaves para diferenciar los dataframes
        df_data["df_name"] = nombre
        df_data["origin"] = "VBF"

        # se guarda el df en la lista
        df_all_signal = pd.concat([df_all_signal, df_data], axis=0)
    
    # se leen los df's introducidos en datasets
    for data in tqdm(ggf_data): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        # añado llaves para diferenciar los dataframes
        df_data["df_name"] = nombre
        df_data["origin"] = "ggF"

        # se guarda el df en la lista
        df_all_background = pd.concat([df_all_background, df_data], axis=0)
    
    df_all = pd.concat([df_all_signal, df_all_background], axis=0)
    df_all.set_index(['origin', 'df_name'], inplace=True)
    return df_all


#Se le da el corte
# se realizan los cortes superiores e inferiores a una lista de dataframes
def do_cuts(df_all, cuts, scale):
    # nombre de la señal a la cual se le mostrará el número de eventos
    name_signal = df_all.index.get_level_values('df_name').unique()[0]

    # se aplican todos los cortes
    for variable in cuts:
        # se grafica el numero de eventos
        numero_eventos_antes = df_all.query('df_name == @name_signal').shape[0]
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

        numero_eventos_despues = df_all.query('df_name == @name_signal').shape[0]
        print(f'Numero eventos antes: {numero_eventos_despues} \n')
    
    # elimino los pesos negativos
    # weights = df_all["intLumi"]*df_all["scale1fb"]
    df_all = df_all[df_all["intLumi"]*df_all["scale1fb"] >= 0]
             
    return df_all