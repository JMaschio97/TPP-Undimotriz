import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

fig = plt.figure(constrained_layout=True, figsize=(15,9))
plt.axis("off")
fig.suptitle(f"\n\nPrueba")
ax = fig.add_subplot(1,1,1)
ax.plot(Prueba['Tiempo'], Prueba["_2"], linewidth=1.5)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
startx, endx = ax.get_xlim()
starty, endy = ax.get_ylim()
ax.set_ylim([starty, endy])
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

fig2 = plt.figure(constrained_layout=True, figsize=(15,9))
plt.axis("off")
fig2.suptitle(f"\n\nMedicion_1")
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(Medicion_1['Tiempo'], Medicion_1["_2"], linewidth=1.5)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
startx, endx = ax2.get_xlim()
starty, endy = ax2.get_ylim()
ax2.set_ylim([starty ,endy])
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.set_xlim(startx, endx)
startx, endx = int(startx), int(endx)
ax2.xaxis.tick_bottom()
ax2.xaxis.set_label_position('bottom')
ax2.set_ylabel("Z_f(t)", rotation=90, fontsize=10.2)
ax2.set_xlabel("t(s)",rotation=90, fontsize=10.2)
ax2.grid(True, zorder=1, alpha=0.5)
plt.tight_layout(pad=2, w_pad=1, h_pad=2)

fig3 = plt.figure(constrained_layout=True, figsize=(15,9))
plt.axis("off")
fig3.suptitle(f"\n\nMedicion_2")
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(Medicion_2['Tiempo'], Medicion_2["_2"], linewidth=1.5)
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
startx, endx = ax3.get_xlim()
starty, endy = ax3.get_ylim()
ax3.set_ylim([starty, endy])
ax3.tick_params(axis="x", labelsize=10)
ax3.tick_params(axis='y', labelsize=10)
ax3.set_xlim(startx, endx)
startx, endx = int(startx), int(endx)
ax3.xaxis.tick_bottom()
ax3.xaxis.set_label_position('bottom')
ax3.set_ylabel("Z_f(t)", rotation=90, fontsize=10.2)
ax3.set_xlabel("t(s)",rotation=90, fontsize=10.2)
ax3.grid(True, zorder=1, alpha=0.5)
plt.tight_layout(pad=2, w_pad=1, h_pad=2)

absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)

nombre_arch = f"Graficos Informe Centrado.pdf"
pp = PdfPages(nombre_arch)


pp.savefig(fig)
pp.savefig(fig2)
pp.savefig(fig3)

pp.close()
