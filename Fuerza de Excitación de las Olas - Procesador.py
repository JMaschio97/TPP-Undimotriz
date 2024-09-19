import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def Graf_Pruebas(Pruebas):
    figuras=[]
    for p in  range(len(Pruebas)):
        df=Pruebas[f'Prueba_{p+1}']
        fig = plt.figure(constrained_layout=True, figsize=(15,9))
        plt.axis("off")
        fig.suptitle(f"\n\nPrueba_{p+1}")
        ax = fig.add_subplot(1,1,1)
        ax.plot(df['Tiempo'], df["_2"], linewidth=1.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        startx, endx = ax.get_xlim()
        starty, endy = ax.get_ylim()
        ax.set_ylim([0, endy])
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(startx, endx)
        startx, endx = int(startx), int(endx)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_ylabel("Z_f(t)", rotation=90, fontsize=10.2)
        ax.set_xlabel("t(s)", fontsize=10.2)
        ax.grid(True, zorder=1, alpha=0.5)
        plt.tight_layout(pad=2, w_pad=1, h_pad=2)
        figuras.append(fig)

    return figuras

def Graf_Mediciones(Pruebas):
    figuras=[]
    for p in  range(len(Pruebas)):
        df=Pruebas[f'Medicion_{p+1}']
        fig = plt.figure(constrained_layout=True, figsize=(15,9))
        plt.axis("off")
        fig.suptitle(f"\n\nMedicion_{p+1}")
        ax = fig.add_subplot(1,1,1)
        ax.plot(df['Tiempo'], df["_2"], linewidth=1.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        startx, endx = ax.get_xlim()
        starty, endy = ax.get_ylim()
        ax.set_ylim([0, endy])
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(startx, endx)
        startx, endx = int(startx), int(endx)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_ylabel("Z_f(t)", rotation=90, fontsize=10.2)
        ax.set_xlabel("t(s)",rotation=90, fontsize=10.2)
        ax.grid(True, zorder=1, alpha=0.5)
        plt.tight_layout(pad=2, w_pad=1, h_pad=2)
        figuras.append(fig)

    return figuras

# Prompt the user to enter the filename (excluding the extension)
filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f"Frecuencia {filename}.xlsx"

# Read all sheets into a dictionary of DataFrames
dfs = pd.read_excel(excel_filename, sheet_name=None)


# Display the DataFrames (one for each sheet)
for sheet_name, df in dfs.items():
    print(f"DataFrame for sheet '{sheet_name}':")
    # Drop the first and last rows from the DataFrame
    # Create the new column 't'
    df['t'] = df.index * 0.1
    mean_F= df['F'].mean()
    mean_H= df['H'].mean()
    df['F']=(df['F']-mean_F)*-0.0221
    df['H']=df['H']-mean_H


    print(df)
    print("\n")



# Define the keys to exclude
keys_to_exclude = ['Estatico D1', 'Estatico D2','Generación de Ola']

# Create a new dictionary excluding the specified keys
dfs_filtered = {key: value for key, value in dfs.items() if key not in keys_to_exclude}





excel_file_path = 'Frecuencia AII_procesada.xlsx'

# Define the output Excel filename
output_excel_filename = "Frecuencia AII_procesada.xlsx"

# Convert the dictionary dfs_filtered to an Excel file
with pd.ExcelWriter(output_excel_filename) as writer:
    for sheet_name, df in dfs_filtered.items():
        # Write each DataFrame to a separate sheet in the Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Iterate through filtered DataFrames and create a DataFrame for each sheet
for sheet_name, df in dfs_filtered.items():
    # Calculate max H values and corresponding t values
    max_H = df['H'].max()
    corresponding_t = df.loc[df['H'].idxmax(), 't']

    # Sort the DataFrame by 't' values
    df.sort_values(by='t', inplace=True)

    # Calculate the difference between consecutive 't' values
    df['Periodo'] = df['t'].diff()

    # Create a new DataFrame containing 't', 'Max_H', and 'Periodo' columns
    df_frecuencia = pd.DataFrame({'t': [corresponding_t], 'Max_H': [max_H]})
    df_frecuencia['Periodo'] = df['Periodo']

    # Save the DataFrame to an Excel file with a specific name
    output_filename = f"Frecuencias {excel_filename}_{sheet_name}.xlsx"
    df_frecuencia.to_excel(output_filename, index=False)
    

###############################Calculo de Frecuencias##################################
    # Después de que has ordenado tu DataFrame por 't'
for sheet_name, df in dfs_filtered.items():
    # Encuentra los índices de los picos máximos locales positivos
    peaks, _ = find_peaks(df['H'], height=0)  # height=0 para considerar solo los máximos positivos
 
    # Selecciona los tiempos y valores de los picos
    peak_times = df.iloc[peaks]['t']
    peak_values = df.iloc[peaks]['H']
 
    # Crea un nuevo DataFrame con los tiempos y valores de los picos
    df_frecuencia = pd.DataFrame({'t': peak_times, 'Max_H': peak_values})
 
    # Calcula los periodos como la diferencia entre los tiempos consecutivos de los picos
    df_frecuencia['Periodo'] = df_frecuencia['t'].diff()
 
    # Calcula la frecuencia como el inverso del período
    # Puedes decidir cómo manejar el primer valor NaN. Una opción es reemplazarlo con el segundo valor de frecuencia o dejarlo como NaN.
    df_frecuencia['Frecuencia'] = 1 / df_frecuencia['Periodo']
    # Opcionalmente, manejar el primer valor NaN de la frecuencia
    # df_frecuencia['Frecuencia'].iloc[0] = df_frecuencia['Frecuencia'].iloc[1]
 
    # Guarda el DataFrame a un archivo Excel con un nombre específico
    output_filename = f"Frecuencias {excel_filename}_{sheet_name}.xlsx"
    df_frecuencia.to_excel(output_filename, index=False)    
