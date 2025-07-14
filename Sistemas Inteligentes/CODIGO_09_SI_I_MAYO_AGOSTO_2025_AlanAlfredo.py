import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Cargar dataset
df = pd.read_csv('fashion-mnist_test.csv')
X = df.drop('label', axis=1).values / 255.0  # Normalizar
y = df['label'].values

# 2. Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo
def crear_modelo(n_neuronas=128, tasa_aprendizaje=0.001, activacion='relu'):
    modelo = keras.models.Sequential([
        keras.layers.Dense(n_neuronas, activation=activacion, input_shape=(784,)),
        keras.layers.Dense(64, activation=activacion),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimizador = keras.optimizers.Adam(learning_rate=tasa_aprendizaje)
    modelo.compile(optimizer=optimizador,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo

# 4. Entrenar varios modelos con diferentes parámetros
resultados = []
parametros = [
    (64, 0.001, 'relu'),
    (128, 0.001, 'relu'),
    (128, 0.005, 'relu'),
    (256, 0.001, 'tanh'),
    (128, 0.0005, 'relu')
]

for n, lr, act in parametros:
    print(f"Entrenando con neuronas={n}, lr={lr}, act={act}")
    modelo = crear_modelo(n, lr, act)
    history = modelo.fit(X_train, y_train, epochs=10, batch_size=32,
                         validation_data=(X_val, y_val), verbose=0)
    y_pred = np.argmax(modelo.predict(X_val), axis=1)
    acc = np.mean(y_pred == y_val)
    ecm = mean_squared_error(y_val, y_pred)
    resultados.append((modelo, acc, ecm, (n, lr, act), history))

# 5. Seleccionar el mejor modelo
mejor_modelo, mejor_acc, mejor_ecm, mejores_parametros, mejor_history = sorted(resultados, key=lambda x: (-x[1], x[2]))[0]
print(f"\nMejor modelo: {mejores_parametros}, Accuracy: {mejor_acc:.4f}, ECM: {mejor_ecm:.4f}")

# 6. Guardar pesos y bias
pesos_bias = mejor_modelo.get_weights()
joblib.dump(pesos_bias, 'pesos_bias_mejor_modelo.pkl')

# 7. Gráficas de desempeño
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(mejor_history.history['accuracy'], label='Entrenamiento')
plt.plot(mejor_history.history['val_accuracy'], label='Validación')
plt.title('Precisión por época')
plt.legend()

plt.subplot(1,2,2)
plt.plot(mejor_history.history['loss'], label='Entrenamiento')
plt.plot(mejor_history.history['val_loss'], label='Validación')
plt.title('Pérdida por época')
plt.legend()

plt.tight_layout()
plt.savefig('grafico_desempeno.png')

# 8. Matriz de confusión
plt.figure(figsize=(8,6))
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('matriz_confusion.png')

# 9. Clasificación detallada
print("\nReporte de clasificación:")
print(classification_report(y_val, y_pred))

