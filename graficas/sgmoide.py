import numpy as np
import matplotlib.pyplot as plt

# Definir la función sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generar valores para el eje x
x = np.linspace(-10, 10, 400)

# Calcular los valores de y utilizando la función sigmoidal
y = sigmoid(x)

# Crear la gráfica
plt.plot(x, y, label='Sigmoid Function')

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()
