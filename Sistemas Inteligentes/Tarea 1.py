# %% 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, 
    precision_recall_curve, classification_report, accuracy_score,
    f1_score, recall_score, precision_score
)
import warnings
warnings.filterwarnings('ignore')

# %% 2. Cargar y preparar datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
columns = ['age', 'year', 'nodes', 'survival_status']
df = pd.read_csv(url, header=None, names=columns)

# Convertir la variable objetivo a binaria
df['survival_status'] = df['survival_status'].map({1: 1, 2: 0})

# Dividir en características (X) y objetivo (y)
X = df.drop('survival_status', axis=1)
y = df['survival_status']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo de regresión logística
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Predecir
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# %% 3. Generar gráficos y tablas (2.4)
# Configurar estilo de gráficos
sns.set_theme(style="whitegrid")

# Gráfico 1: Distribución de la variable objetivo (Figura 1)
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Distribución de Sobrevivencia (1: Sí, 0: No)')
plt.savefig('target_distribution.png', bbox_inches='tight')
plt.close()

# Gráfico 2: Boxplots de características vs. objetivo (Figura 2)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(X.columns):
    sns.boxplot(x=y, y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')
plt.savefig('feature_boxplots.png', bbox_inches='tight')
plt.close()

# Gráfico 3: Matriz de correlación (Figura 3)
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.close()

# Gráfico 4: Matriz de confusión (Figura 4)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Gráfico 5: Curva ROC (Figura 5)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.savefig('roc_curve.png', bbox_inches='tight')
plt.close()

# Gráfico 6: Curva Precisión-Recall (Figura 6)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Regresión Logística')
plt.title('Curva Precisión-Recall')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.legend()
plt.savefig('precision_recall_curve.png', bbox_inches='tight')
plt.close()

# Gráfico 7: Coeficientes del modelo (Figura 7)
coefs = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(8, 4))
coefs.plot(kind='bar')
plt.title('Coeficientes de la Regresión Logística')
plt.savefig('coefficients.png', bbox_inches='tight')
plt.close()

# Gráfico 8: Distribución de probabilidades predichas (Figura 8)
plt.figure(figsize=(8, 4))
sns.histplot(y_proba, bins=20, kde=True)
plt.title('Distribución de Probabilidades Predichas')
plt.xlabel('Probabilidad de Sobrevivencia')
plt.savefig('predicted_probabilities.png', bbox_inches='tight')
plt.close()

# Tabla 1: Reporte de clasificación (Tabla 1)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv')

# Tabla 2: Métricas clave (Tabla 2)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precisión (Clase 0)': report['0']['precision'],
    'Recall (Clase 0)': report['0']['recall'],
    'Precisión (Clase 1)': report['1']['precision'],
    'Recall (Clase 1)': report['1']['recall'],
    'F1-Score (Global)': f1_score(y_test, y_pred)
}
pd.DataFrame(metrics, index=['Valor']).transpose().to_csv('key_metrics.csv')