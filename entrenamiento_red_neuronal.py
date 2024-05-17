import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix   
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.saving import save_model

dividir = pd.read_csv("C:/Users/drngb/Desktop/tfg/prueba/prueba_10000.csv")

X = dividir.drop(columns=['ataque_normal', 'ataque_attack', 'ip.src', 'ip.dst'])
y = dividir['ataque_attack']

# entrenamiento (80%) prueba (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# entrenamiento (75%) validación (25%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64, activation='relu')) 
model.add(Dense(units=1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con datos de entrenamiento y evaluar en datos de validación
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

save_model(model, 'modelo_red_neuronal.keras')

# Visualizar el rendimiento del modelo en los datos de entrenamiento y validación
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predecir las probabilidades para las clases
y_pred_proba = model.predict(X_test)


y_pred = (y_pred_proba > 0.5).astype(int)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir las métricas de evaluación
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
