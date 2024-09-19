import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")


def calculate_peaks(data):
     peaks, _ = find_peaks(data, height=0)  # Assuming only positive peaks are of interest
     peaks_negative, _ = find_peaks(-data, prominence=0)  # Finding negative peaks
     negative_peaks = data[peaks_negative]
     positive_peaks = data[peaks]
     mean_peaks =  np.mean(positive_peaks)-np.mean(negative_peaks)
     return mean_peaks
# Prompt the user to enter the filename (excluding the extension)

# Prompt the user to enter the filename (excluding the extension)
filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f"Frecuencia {filename}.xlsx"

df = pd.read_excel(excel_filename)

# Rename the columns


# Now the columns are renamed according to your specified names
# Calculate the values for the 't' column based on the index
df['t'] = df.index * 0.052
tiempo = df['t'].values
# Calculate the derivative and save it in a new column 'r_z'
df['r_z'] = df['a_z'].diff()
mean_F= df['F'].mean()
mean_H= df['d'].mean()
df['F']=(df['F']-mean_F)*-0.0221*9.81
df['d']=(df['d']-mean_H)/1000

# Especifica la ruta donde deseas guardar el archivo de Excel
ruta_excel = f'Frecuencia_{filename}.xlsx'

# Guarda el DataFrame en un archivo de Excel
df.to_excel(ruta_excel, index=False)

print("¡DataFrame convertido a Excel exitosamente!")

datos_H = df['d'].values  # Asume  es tu canal de Altura de Ola
datos_F = df['F'].values  # Asume  es tu canal de Fuerzas
mean_positive_peaks_F = calculate_peaks(datos_F)
mean_positive_peaks_H = calculate_peaks(datos_H)

# Now the derivative of 'a_z' is calculated and stored in the new column 'r_z'
# Calculate the Constants
Dens_agua=998
g=9.81
R=0.25
A_w =0.19635
K=1922.33662
# Prompt the user to enter the filename (excluding the extension)
# Calcular la transformada de Fourier de H
fft_datos_H = np.fft.fft(datos_H)
frecuencias_H = np.fft.fftfreq(len(datos_H), d=tiempo[1] - tiempo[0])  # d es el intervalo de muestreo
# Calcular la transformada de Fourier de F
fft_datos_F = np.fft.fft(datos_F)
frecuencias_F = np.fft.fftfreq(len(datos_F), d=tiempo[1] - tiempo[0])  # d es el intervalo de muestreo

# Tomar solo la mitad positiva del espectro de H
mitad_H = len(frecuencias_H) // 2
fft_positivo_H = np.abs(fft_datos_H[:mitad_H])
frecuencias_positivas_H = frecuencias_H[:mitad_H]
# Tomar solo la mitad positiva del espectro de F
mitad_F = len(frecuencias_F) // 2
fft_positivo_F = np.abs(fft_datos_F[:mitad_F])
frecuencias_positivas_F = frecuencias_F[:mitad_F]

# Encontrar la frecuencia con la amplitud máxima de H, ignorando el componente DC
pico_index_H = np.argmax(fft_positivo_H[1:]) + 1  # '+1' para compensar el índice que ignoramos
frecuencia_natural_H = frecuencias_positivas_H[pico_index_H]
# Encontrar la frecuencia con la amplitud máxima de F, ignorando el componente DC
pico_index_F = np.argmax(fft_positivo_F[1:]) + 1  # '+1' para compensar el índice que ignoramos
frecuencia_natural_F = frecuencias_positivas_F[pico_index_F]
print("Esta es la Frecuencia F:")
print(frecuencia_natural_F)
print("Esta es la Frecuencia H:")
print(frecuencia_natural_H)
# m = float(input("Ingrese la Masa: "))
# f= float(input("Ingrese la Frecuencia: "))
# w=float(2*3.1415*f)

# m_a=((K-(mean_positive_peaks_F/mean_positive_peaks_H))/(9.8696))-m
# b_h=(mean_positive_peaks_F/mean_positive_peaks_H)/3.1415


# PDF file to save plots
pdf_filename = f"Analysis_{filename}.pdf"
pdf_pages = PdfPages(pdf_filename)
# Graficar el espectro de frecuencia de Ola
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_positivas_H, fft_positivo_H, label='Espectro de frecuencia')
plt.plot(frecuencia_natural_H, fft_positivo_H[pico_index_H], 'ro', label=f'Frecuencia: {frecuencia_natural_H:.2f} Hz')
plt.title(f'Análisis de la Transformada de Fourier de Ola - {filename}')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud H')
plt.legend()
plt.grid(True)
pdf_pages.savefig()


print(f"La frecuencia estimada de Ola de {filename}  es: {frecuencia_natural_H:.2f} Hz")

# Graficar el espectro de frecuencia de fuerza
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_positivas_F, fft_positivo_F, label='Espectro de frecuencia')
plt.plot(frecuencia_natural_F, fft_positivo_F[pico_index_F], 'ro', label=f'Frecuencia: {frecuencia_natural_F:.2f} Hz')
plt.title(f'Análisis de la Transformada de Fourier de Fuerzas - {filename}')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud F')
plt.legend()
plt.grid(True)
pdf_pages.savefig()

# Plot F vs t
plt.figure(figsize=(10, 6))
plt.plot(tiempo, datos_F, label='F vs t')
plt.title(f'F vs t - {filename}')
plt.xlabel('Tiempo')
plt.ylabel('F')
plt.legend()
plt.grid(True)
pdf_pages.savefig()
plt.close()

# Plot H vs t
plt.figure(figsize=(10, 6))
plt.plot(tiempo, datos_H, label='H vs t')
plt.title(f'H vs t - {filename}')
plt.xlabel('Tiempo')
plt.ylabel('H')
plt.legend()
plt.grid(True)
pdf_pages.savefig()
plt.close()

# Plot r_z vs t
plt.figure(figsize=(10, 6))
plt.plot(tiempo, df['r_z'], label='r_z vs t')
plt.title(f'r_z vs t - {filename}')
plt.xlabel('Tiempo')
plt.ylabel('r_z')
plt.legend()
plt.grid(True)
pdf_pages.savefig()
plt.close()

# Close the PDF file
pdf_pages.close()


# # List to store dominant frequencies for each sheet
# dominant_frequencies = []
# dominant_frequencies.append({'sheet_name': filename,
#                                 'f': f,
#                                 'w': w,
#                                 'mean_peak_F': mean_positive_peaks_F,
#                                 'mean_peak_H': mean_positive_peaks_H,
#                                 'K':K,
#                                 'm_a':m_a,
#                                 'b_h':b_h})

# # Convert the list to a DataFrame
# df_dominant_frequencies = pd.DataFrame(dominant_frequencies)
# # Especifica la ruta donde deseas guardar el archivo de Excel
# ruta_excel = f'Calculos {filename}.xlsx'

# # Guarda el DataFrame en un archivo de Excel
# df_dominant_frequencies.to_excel(ruta_excel, index=False)

# print("¡DataFrame convertido a Excel exitosamente!")
