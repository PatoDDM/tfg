import numpy as np
import matplotlib.pyplot as plt

# Definir la función ReLU
def relu(x):
    return np.maximum(0, x)

# Generar valores para el eje x
x = np.linspace(-10, 10, 400)

# Calcular los valores de y utilizando la función ReLU
y = relu(x)

# Crear la gráfica
plt.plot(x, y, label='ReLU Function')

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.title('ReLU Function')
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()
