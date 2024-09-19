import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

# Prompt the user to enter the filename (excluding the extension)
filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f'Frecuencia_{filename}.xlsx'

df = pd.read_excel(excel_filename)

tiempo = df['t'].values
datos_H = df['d'].values  # Asume  es tu canal de Altura de Ola
datos_F = df['F'].values  # Asume  es tu canal de Fuerzas

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

# Aplica el filtro en el dominio de la frecuencia para las señales H y F
filtro_H = (np.abs(frecuencias_H) <= 0.7) & (np.abs(frecuencias_H) >= 0.4)
filtro_F = (np.abs(frecuencias_F) <= 0.7) & (np.abs(frecuencias_F) >= 0.4)

fft_datos_H_filtrada = fft_datos_H.copy()
fft_datos_F_filtrada = fft_datos_F.copy()

fft_datos_H_filtrada[~filtro_H] = 0
fft_datos_F_filtrada[~filtro_F] = 0

# Aplica la antitransformada de Fourier para obtener la señal original filtrada
datos_H_filtrada = np.fft.ifft(fft_datos_H_filtrada)
datos_F_filtrada = np.fft.ifft(fft_datos_F_filtrada)

# Puedes guardar los datos filtrados en tu DataFrame si lo deseas
df['d_filtrada'] = datos_H_filtrada.real  # Solo toma la parte real ya que puede haber pequeñas partes imaginarias debido a errores numéricos
df['F_filtrada'] = datos_F_filtrada.real  # Solo toma la parte real ya que puede haber pequeñas partes imaginarias debido a errores numéricos

# Especifica la ruta donde deseas guardar el archivo de Excel
ruta_excel = f'Frecuencia_{filename}_filtrada.xlsx'

# Guarda el DataFrame en un archivo de Excel
df.to_excel(ruta_excel, index=False)

# PDF file to save plots
pdf_filename = f"Analysis_{filename}_filtrada.pdf"
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

# Plot F_filtrada vs t
plt.figure(figsize=(10, 6))
plt.plot(tiempo, df['F_filtrada'], label='F vs t')
plt.title(f'F_filtrada vs t - {filename}')
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

# Plot H_filtrada vs t
plt.figure(figsize=(10, 6))
plt.plot(tiempo, df['d_filtrada'], label='H vs t')
plt.title(f'H_filtrada vs t - {filename}')
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


# Extrae las señales filtradas de H y F
datos_H_filtrada = df['d_filtrada'].values
datos_F_filtrada = df['F_filtrada'].values

# Calcular la transformada de Fourier de H
fft_datos_H = np.fft.fft(datos_H_filtrada)
frecuencias_H = np.fft.fftfreq(len(datos_H_filtrada), d=tiempo[1] - tiempo[0])  # d es el intervalo de muestreo
# Calcular la transformada de Fourier de F
fft_datos_F = np.fft.fft(datos_F_filtrada)
frecuencias_F = np.fft.fftfreq(len(datos_F_filtrada), d=tiempo[1] - tiempo[0])  # d es el intervalo de muestreo

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
