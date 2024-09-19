import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

# Prompt the user to enter the filename (excluding the extension)
filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f'Frecuencia_{filename}_filtrada.xlsx'

df = pd.read_excel(excel_filename)



# Obtener las señales como arrays
F = df["F_filtrada"].values
d = df["d_filtrada"].values
t = df["t"].values



# Definir la función de ajuste, en este caso una señal senoidal con un desfase
def func(t, A, omega, phi):
    return A * np.sin(omega * t + phi)

# Realizar el ajuste de curvas
params_F, params_covariance_F = curve_fit(func, t, F)
params_d, params_covariance_d = curve_fit(func, t, d)

# Extraer los parámetros del ajuste
A_F, omega_F, phi_F = params_F
A_d, omega_d, phi_d = params_d

# Calcular el desfase entre las señales
desfase = phi_d - phi_F

# Asegurar que el desfase está en el rango [-pi, pi]
desfase = ((desfase + np.pi) % (2 * np.pi)) - np.pi

print("Ángulo de desfase (radianes):", desfase)
print("Ángulo de desfase (grados):", np.degrees(desfase))

Integrador.py:
mport pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapz
warnings.filterwarnings("ignore")

def calculate_peaks(data):
     peaks, _ = find_peaks(data, height=0)  # Assuming only positive peaks are of interest
     peaks_negative, _ = find_peaks(-data, prominence=0)  # Finding negative peaks
     negative_peaks = data[peaks_negative]
     positive_peaks = data[peaks]
     mean_peaks =  np.mean(positive_peaks)-np.mean(negative_peaks)
     return mean_peaks


# Prompt the user to enter the filename (excluding the extension)
filename = input("Ingrese la Frecuencia: ")

# Concatenate the user's input with the rest of the filename string
excel_filename = f'Frecuencia_{filename}_filtrada.xlsx'

df = pd.read_excel(excel_filename)

# mean_positive_peaks_H =calculate_peaks(df['d_filtrada'])
mean_positive_peaks_H = np.sqrt(np.mean(df['d_filtrada']**2))*np.sqrt(2)
# mean_positive_peaks_H = np.sqrt(np.mean(df['d_filtrada']**2))
F_mean=np.sqrt(np.mean(df['F_filtrada']**2))*np.sqrt(2)
# F_mean=np.sqrt(np.mean(df['F_filtrada']**2))
# F_mean=calculate_peaks(df['F_filtrada'])
print("F_mean",F_mean)
omega = 2 * np.pi*0.5
e_fz=0.10768







# F_sin =np.abs(F_mean*np.sin(e_fz))

F_sin =F_mean*np.sin(e_fz)


# Calcular la integral numéricamente utilizando la regla del trapecio
# F_cos = np.abs(F_mean*np.cos(e_fz))

F_cos = F_mean*np.cos(e_fz)


print("F_sin:", F_sin)


print("F_cos:", F_cos)

Dens_agua=998
g=9.81
R=0.25
A_w =0.19635
K=1922.33662

m = float(input("Ingrese la Masa: "))
f= float(input("Ingrese la Frecuencia: "))
w=float(2*3.1415*f)

m_a=((K-(F_cos/mean_positive_peaks_H))/(9.869))-m
b_h=(F_sin/mean_positive_peaks_H)/(2*np.pi*0.5)

print("m_a=",m_a)

print("b_h=",b_h)
# List to store dominant frequencies for each sheet
dominant_frequencies = []
dominant_frequencies.append({'sheet_name': filename,
                                'f': f,
                                'w': w,
                                'F_sin': F_sin,
                                'F_cos': F_cos,
                                'mean_peak_H': mean_positive_peaks_H,
                                'K':K,
                                'm_a':m_a,
                                'b_h':b_h})

# Convert the list to a DataFrame
df_dominant_frequencies = pd.DataFrame(dominant_frequencies)
# Especifica la ruta donde deseas guardar el archivo de Excel
ruta_excel = f'Calculos {filename}_filtrada_DEF.xlsx'

# Guarda el DataFrame en un archivo de Excel
df_dominant_frequencies.to_excel(ruta_excel, index=False)

