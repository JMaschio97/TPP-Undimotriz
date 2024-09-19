import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)
print(path)
print(filename)
ubicacion = path + "\Prueba.xlsx"

Prueba = pd.read_excel(ubicacion, index_col=None, header=0, usecols=('_2', 'Tiempo'))
print(Prueba)
Prueba["_2"]=Prueba["_2"]-169.6

absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)
print(path)
print(filename)
ubicacion = path + "\Medicion_1.xlsx"

Medicion_1= pd.read_excel(ubicacion, index_col=None, header=0, usecols=('_2', 'Tiempo'))
print(Medicion_1)
Medicion_1["_2"]=Medicion_1["_2"]-171.9

absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)
print(path)
print(filename)
ubicacion = path + "\Medicion_2.xlsx"

Medicion_2= pd.read_excel(ubicacion, index_col=None, header=0, usecols=('_2', 'Tiempo'))
print(Medicion_2)
Medicion_2["_2"]=Medicion_2["_2"]-167.8

Prueba = pd.read_excel(ubicacion, index_col=None, header=0, usecols=('_2', 'Tiempo'))
# Supongamos que tus datos están en un DataFrame llamado Prueba
# Si tus datos están en un archivo, puedes leerlos así:

 
# Asegúrate de que la columna 'Tiempo' y el canal de interés ('_2') estén en el formato correcto
tiempo = Medicion_1['Tiempo'].values
datos = Medicion_1['_2'].values  # Asume que '_2' es tu canal de interés
 
# Calcular la transformada de Fourier
fft_datos = np.fft.fft(datos)
frecuencias = np.fft.fftfreq(len(datos), d=tiempo[1] - tiempo[0])  # d es el intervalo de muestreo
 
# Tomar solo la mitad positiva del espectro
mitad = len(frecuencias) // 2
fft_positivo = np.abs(fft_datos[:mitad])
frecuencias_positivas = frecuencias[:mitad]
 
# Encontrar la frecuencia con la amplitud máxima, ignorando el componente DC
pico_index = np.argmax(fft_positivo[1:]) + 1  # '+1' para compensar el índice que ignoramos
frecuencia_natural = frecuencias_positivas[pico_index]
 
# Graficar el espectro de frecuencia
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_positivas, fft_positivo, label='Espectro de frecuencia')
plt.plot(frecuencia_natural, fft_positivo[pico_index], 'ro', label=f'Frecuencia natural: {frecuencia_natural:.2f} Hz')
plt.title('Análisis de la Transformada de Fourier - Medicion 2')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()
 
print(f"La frecuencia natural estimada es: {frecuencia_natural:.2f} Hz")
