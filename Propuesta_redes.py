import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Cargar datos
# --------------------------------------------------

datos = pd.read_csv("Redes Neuronales/2do Parcial/Programas/yeast.data", sep=r"\s+", header=None)

nombres_columnas = [
    "sequence_name",
    "mcg", "gvh", "alm", "mit", "erl", "pox", "vac",
    "nuc", "class"
]

datos.columns = nombres_columnas

datos = datos.drop("sequence_name", axis=1)

atributos = datos.drop("class", axis=1)
etiquetas = datos["class"]

codificador = LabelEncoder()
etiquetas_codificadas = codificador.fit_transform(etiquetas)

escala = StandardScaler()
atributos_escalados = escala.fit_transform(atributos)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    atributos_escalados, etiquetas_codificadas, test_size=0.2, random_state=42
)

X_entrenamiento = torch.tensor(X_entrenamiento, dtype=torch.float32)
X_prueba = torch.tensor(X_prueba, dtype=torch.float32)
y_entrenamiento = torch.tensor(y_entrenamiento, dtype=torch.long)
y_prueba = torch.tensor(y_prueba, dtype=torch.long)

# Crear dataset y dataloader
dataset = TensorDataset(X_entrenamiento, y_entrenamiento)
cargador = DataLoader(dataset, batch_size=32, shuffle=True)

# --------------------------------------------------
# 2. Modelo más preciso
# --------------------------------------------------

class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()

        self.capa1 = nn.Linear(8, 256)
        self.capa2 = nn.Linear(256, 128)
        self.capa3 = nn.Linear(128, 64)
        self.capa_salida = nn.Linear(64, 10)

        self.activacion = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.activacion(self.capa1(x))
        x = self.dropout(x)

        x = self.activacion(self.capa2(x))
        x = self.dropout(x)

        x = self.activacion(self.capa3(x))
        x = self.dropout(x)

        x = self.capa_salida(x)
        return x

modelo = RedNeuronal()

# --------------------------------------------------
# 3. Entrenamiento con Early Stopping
# --------------------------------------------------

criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.0005)

epocas = 200
paciencia = 15
mejor_perdida = float("inf")
espera = 0

lista_perdida = []
lista_exactitud = []

for epoca in range(epocas):

    modelo.train()
    perdida_total = 0

    for lote_X, lote_y in cargador:
        optimizador.zero_grad()
        salida = modelo(lote_X)
        perdida = criterio(salida, lote_y)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()

    perdida_promedio = perdida_total / len(cargador)

    modelo.eval()
    with torch.no_grad():
        salida_entrenamiento = modelo(X_entrenamiento)
        pred = torch.argmax(salida_entrenamiento, dim=1)
        exactitud = accuracy_score(y_entrenamiento.numpy(), pred.numpy())

    lista_perdida.append(perdida_promedio)
    lista_exactitud.append(exactitud)

    print("Época:", epoca+1, ", Pérdida:", perdida_promedio, ", Exactitud:", exactitud)

    if perdida_promedio < mejor_perdida:
        mejor_perdida = perdida_promedio
        espera = 0
        mejor_modelo = modelo.state_dict()
    else:
        espera += 1
        if espera >= paciencia:
            print("Deteniendo entrenamiento por early stopping...")
            break

modelo.load_state_dict(mejor_modelo)

# --------------------------------------------------
# 4. Evaluación final
# --------------------------------------------------

salida_prueba = modelo(X_prueba)
pred_final = torch.argmax(salida_prueba, dim=1)

exactitud_final = accuracy_score(y_prueba.numpy(), pred_final.numpy())

print("")
print("Exactitud final del modelo:", exactitud_final)

# --------------------------------------------------
# 5. Gráficas
# --------------------------------------------------

plt.figure()
plt.plot(lista_perdida)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

plt.figure()
plt.plot(lista_exactitud)
plt.title("Exactitud durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Exactitud")
plt.show()