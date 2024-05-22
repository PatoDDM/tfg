import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Cargar el dataset desde el archivo CSV
df = pd.read_csv('C:/Users/drngb/Desktop/tfg/prueba/prueba_10000.csv')
X = df.drop(columns=['ataque'])
y = df['ataque']

# Asegurarse de que las características son numéricas (convertir si es necesario)
X = pd.get_dummies(X)

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Crear la gráfica t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pd.Categorical(y).codes, cmap='tab10', alpha=0.7)

# Añadir una barra de colores
legend = plt.legend(*scatter.legend_elements(), title="Ataque")
plt.gca().add_artist(legend)

# Añadir etiquetas y título
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Attack Dataset')

# Mostrar la gráfica
plt.grid(True)
plt.show()
