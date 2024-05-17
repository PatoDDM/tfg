import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix   
import matplotlib.pyplot as plt
import joblib

# Cargar los datos
dividir = pd.read_csv("C:/Users/drngb/Desktop/tfg/prueba/prueba_10000.csv")
X = dividir.drop(columns=['ataque_normal', 'ataque_attack', 'ip.src', 'ip.dst'])
y = dividir['ataque_attack']

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dividir los datos de entrenamiento en entrenamiento (75%) y validación (25%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_random_forest.pkl')

# Evaluar el modelo en el conjunto de validación
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Evaluar el modelo en el conjunto de prueba
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Imprimir las métricas de evaluación
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)
print("Test Confusion Matrix:")
print(conf_matrix)

# Visualizar la importancia de las características
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.xticks(rotation=90)
plt.show()
