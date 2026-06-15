import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8, 6.8, 4.3, 7.5, 3.6, 5.2, 8.1, 1.9, 3.7, 5.9, 2.2, 6.5, 7.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9, 6.6, 4.2, 7.0, 4.1, 5.8, 7.3, 2.2, 3.8, 5.5, 3.0, 6.0, 7.1],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3, 5, 3, 5, 2, 3, 6, 1, 3, 4, 2, 5, 5],
    'MYC_expr': [1.5, 4.8, 1.2, 3.9, 5.1, 2.0, 4.9, 1.8, 3.8, 2.4, 4.5, 2.2, 3.1, 4.7, 1.3, 2.5, 3.7, 1.6, 4.1, 5.0],
    'CDKN2A_expr': [0.8, 2.3, 0.6, 1.7, 2.5, 1.0, 2.4, 0.9, 1.8, 1.2, 2.1, 1.1, 1.6, 2.2, 0.7, 1.3, 1.9, 0.8, 2.0, 2.6],
    'Promoter_methylation': [70, 20, 85, 30, 25, 65, 15, 75, 35, 60, 18, 55, 40, 22, 80, 58, 33, 68, 28, 19],
    'Chromatin_accessibility': [0.35, 0.92, 0.25, 0.78, 0.89, 0.42, 0.94, 0.31, 0.74, 0.45, 0.90, 0.50, 0.63, 0.91, 0.22, 0.47, 0.70, 0.38, 0.81, 0.93],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop(columns='Cancer_status')
y = df['Cancer_status']

# WARIANT 1 - ORYGINALNY
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='relu',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n==================== WARIANT 1 ====================")
print("Podział: 75% trening / 25% test")
print("Neurony: (8,4)")
print("Aktywacja: relu")
print("Iteracje: 1000")
print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Zdrowy', 'Rak'],
    yticklabels=['Zdrowy', 'Rak']
)
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek - Wariant 1')
plt.tight_layout()
plt.show()

# WARIANT 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(12, 6),
    activation='relu',
    max_iter=2000,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n==================== WARIANT 2 ====================")
print("Podział: 60% trening / 40% test")
print("Neurony: (12,6)")
print("Aktywacja: relu")
print("Iteracje: 2000")
print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Zdrowy', 'Rak'],
    yticklabels=['Zdrowy', 'Rak']
)
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek - Wariant 2')
plt.tight_layout()
plt.show()

# WARIANT 3
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation='tanh',
    max_iter=1500,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n==================== WARIANT 3 ====================")
print("Podział: 70% trening / 30% test")
print("Neurony: (10,5)")
print("Aktywacja: tanh")
print("Iteracje: 1500")
print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Zdrowy', 'Rak'],
    yticklabels=['Zdrowy', 'Rak']
)
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek - Wariant 3')
plt.tight_layout()
plt.show()

# Komentarz
# Dla wszystkich trzech wariantów uzyskano identyczne wyniki:
# Accuracy = 1.00
# Precision = 1.00
# Recall = 1.00
# F1-score = 1.00
#
# Oznacza to, że model poprawnie sklasyfikował wszystkie próbki
# znajdujące się w zbiorach testowych. Zmiana liczby neuronów,
# liczby iteracji, funkcji aktywacji oraz podziału danych nie
# wpłynęła na wartości metryk.
#
# Macierze pomyłek dla wszystkich wariantów zawierały wyłącznie
# poprawne klasyfikacje i nie wystąpiły błędy klasyfikacji.
#
# Uzyskana wysoka skuteczność i poprawność może być spowodowana
# wyraźnymi podzialami między zbiorami oraz małą liczbą danych,
# co przekłada się na łatwiejszą analizę dla modelu.
