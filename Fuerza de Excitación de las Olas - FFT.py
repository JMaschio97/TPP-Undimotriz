import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks


def calculate_peaks(data):
     peaks, _ = find_peaks(data, height=0)  # Assuming only positive peaks are of interest
     peaks_negative, _ = find_peaks(-data, prominence=0)  # Finding negative peaks
     negative_peaks = data[peaks_negative]
     positive_peaks = data[peaks]
     mean_peaks =  np.mean(positive_peaks)-np.mean(negative_peaks)
     return mean_peaks
# Prompt the user to enter the filename (excluding the extension)

def calculate_mean_peak_difference(data):
    peaks, _ = find_peaks(data)
    all_peaks = data[peaks]

    # Initialize a list to store peak differences
    peak_diffs = []

    # Calculate differences between consecutive peaks
    for i in range(len(all_peaks) - 1):
        peak_diff = all_peaks[i + 1] - all_peaks[i]
        peak_diffs.append(peak_diff)

    # Calculate the mean of peak differences
    mean_peak_diff = sum(peak_diffs) / len(peak_diffs)
    
    return mean_peak_diff

filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f"Frecuencia {filename}_procesada.xlsx"

# Read all sheets into a dictionary of DataFrames
dfs = pd.read_excel(excel_filename, sheet_name=None)

# List to store dominant frequencies for each sheet
dominant_frequencies = []

# PDF file to save plots
pdf_filename = f"Wave_Analysis_{filename}.pdf"
pdf_pages = PdfPages(pdf_filename)


# Display the DataFrames (one for each sheet)
for sheet_name, df in dfs.items():
    print(f"DataFrame for sheet '{sheet_name}':")

    # Asegúrate de que la columna 'Tiempo' y el canal de interés estén en el formato correcto
    tiempo = df['t'].values
    datos_H = df['H'].values  # Asume  es tu canal de Altura de Ola
    datos_F = df['F'].values  # Asume  es tu canal de Fuerzas
    #  Calculate the mean of positive peaks of 'F'
    mean_positive_peaks_F = calculate_peaks(datos_F)
    mean_positive_peaks_H = calculate_peaks(datos_H)
    # Calculate the mean of differences between all consecutive peaks for 'F'
    mean_peak_diff_F = calculate_mean_peak_difference(datos_F)
    mean_peak_diff_H = calculate_mean_peak_difference(datos_H)

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

    # # Store dominant frequencies and mean of positive peaks of 'F'
    # dominant_frequencies.append({'sheet_name': sheet_name,
    #                               'frecuencia_natural_F': frecuencia_natural_F,
    #                               'frecuencia_natural_H': frecuencia_natural_H,
    #                               'mean_positive_peaks_F': mean_positive_peaks_F})
    # Store dominant frequencies and mean of peak differences for 'F'
    dominant_frequencies.append({'sheet_name': sheet_name,
                                  'frecuencia_natural_F': frecuencia_natural_F,
                                  'frecuencia_natural_H': frecuencia_natural_H,
                                  'mean_peak_diff_F': mean_peak_diff_F,
                                  'mean_peak_diff_H': mean_peak_diff_H,
                                  'mean_peak_positive_F': mean_positive_peaks_F,
                                  'mean_peak_positive_H': mean_positive_peaks_H})

    

    # Graficar el espectro de frecuencia de Ola
    plt.figure(figsize=(10, 6))
    plt.plot(frecuencias_positivas_H, fft_positivo_H, label='Espectro de frecuencia')
    plt.plot(frecuencia_natural_H, fft_positivo_H[pico_index_H], 'ro', label=f'Frecuencia: {frecuencia_natural_H:.2f} Hz')
    plt.title(f'Análisis de la Transformada de Fourier de Ola - {sheet_name}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud H')
    plt.legend()
    plt.grid(True)
    pdf_pages.savefig()
   

    print(f"La frecuencia estimada de Ola de {sheet_name}  es: {frecuencia_natural_H:.2f} Hz")

    # Graficar el espectro de frecuencia de fuerza
    plt.figure(figsize=(10, 6))
    plt.plot(frecuencias_positivas_F, fft_positivo_F, label='Espectro de frecuencia')
    plt.plot(frecuencia_natural_F, fft_positivo_F[pico_index_F], 'ro', label=f'Frecuencia: {frecuencia_natural_F:.2f} Hz')
    plt.title(f'Análisis de la Transformada de Fourier de Fuerzas - {sheet_name}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud F')
    plt.legend()
    plt.grid(True)
    pdf_pages.savefig()

    # Plot F vs t
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo, datos_F, label='F vs t')
    plt.title(f'F vs t - {sheet_name}')
    plt.xlabel('Tiempo')
    plt.ylabel('F')
    plt.legend()
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    # Plot H vs t
    plt.figure(figsize=(10, 6))
    plt.plot(tiempo, datos_H, label='H vs t')
    plt.title(f'H vs t - {sheet_name}')
    plt.xlabel('Tiempo')
    plt.ylabel('H')
    plt.legend()
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()
  

    print(f"La frecuencia estimada de Fuerzas de {sheet_name}  es: {frecuencia_natural_F:.2f} Hz")



    print(df)
    print("\n")


# Close the PDF file
pdf_pages.close()

# Save dominant frequencies to Excel
df_dominant_frequencies = pd.DataFrame(dominant_frequencies)
excel_output_filename = f"dominant_frequencies_{filename}.xlsx"
df_dominant_frequencies.to_excel(excel_output_filename, index=False)

print("Plots saved to PDF and dominant frequencies saved to Excel.")
