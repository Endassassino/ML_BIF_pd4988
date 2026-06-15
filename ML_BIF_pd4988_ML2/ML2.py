import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
    'KRAS': [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, zero_division=0))

# KOMENTARZ:
# OCENA MODELU:
#
# Uzyskane wartości metryk:
# Accuracy = 1.00
# Precision = 1.00
# Recall = 1.00
# F1-score = 1.00
#
# Model poprawnie sklasyfikował wszystkie próbki ze zbioru
# testowego, osiągając 100% skuteczności. Oznacza to, że nie
# wystąpiły ani błędy fałszywie dodatnie, ani fałszywie ujemne.
#
# F1-score jest średnią harmoniczną precision i recall, i
# ze względu na bardzo mały zbiór danych wyniki mogą
# być niestabilne i silnie zależeć od losowego podziału danych.
# W praktyce do oceny modelu należałoby wykorzystać znacznie większy
# zbiór danych lub walidację krzyżową.

