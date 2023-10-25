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
    df_all_VBF = pd.DataFrame()
    df_all_ggF = pd.DataFrame()

    # se leen los df's introducidos en datasets
    for data in tqdm(vbf_data): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe
        print(nombre)

        # añado llaves para diferenciar los dataframes
        df_data["df_name"] = nombre
        df_data["origin"] = "VBF"

        # se guarda el df en la lista
        df_all_VBF = pd.concat([df_all_VBF, df_data], axis=0)
    
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
        df_all_ggF = pd.concat([df_all_ggF, df_data], axis=0)
    
    df_all = pd.concat([df_all_VBF, df_all_ggF], axis=0)
    df_all.set_index(['origin', 'df_name'], inplace=True)
    return df_all


