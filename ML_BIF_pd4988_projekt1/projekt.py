import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("dane_projekt1.csv")

print("=" * 60)
print("PIERWSZE 5 WIERSZY")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("INFORMACJE O DANYCH")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("STATYSTYKI OPISOWE")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("BRAKUJĄCE WARTOŚCI")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("ROZKŁAD KLAS")
print("=" * 60)
print(df["Gene_Function"].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Gene_Function")
plt.title("Liczba genów w poszczególnych klasach")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

X = df.drop(columns=["Gene_ID", "Gene_Function"])

y = df["Gene_Function"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nMapowanie klas:")

for i, name in enumerate(encoder.classes_):
    print(f"{name} -> {i}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nRozmiar zbioru treningowego:", X_train.shape)
print("Rozmiar zbioru testowego:", X_test.shape)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("WYNIKI MODELU")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")

print("\nRaport klasyfikacji:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Macierz pomyłek")
plt.xlabel("Przewidziana klasa")
plt.ylabel("Prawdziwa klasa")
plt.tight_layout()
plt.show()

importance = pd.DataFrame({
    "Cecha": X.columns,
    "Ważność": rf.feature_importances_
})

importance = importance.sort_values(
    by="Ważność",
    ascending=False
)

print("\n" + "=" * 60)
print("WAŻNOŚĆ CECH")
print("=" * 60)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Ważność",
    y="Cecha"
)

plt.title("Ważność cech w modelu Random Forest")
plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("INTERPRETACJA")
print("=" * 60)

print(f"""
Dokładność modelu wynosi {accuracy:.2%}.

Random Forest został wybrany ponieważ:
- dobrze działa dla klasyfikacji wieloklasowej,
- radzi sobie z małymi zbiorami danych,
- umożliwia analizę ważności cech,
- nie wymaga skalowania danych.

Jeżeli accuracy jest niskie, może to wynikać z:
- małej liczby obserwacji,
- syntetycznego charakteru danych,
- braku silnego związku pomiędzy ekspresją genów
  a przypisaną funkcją biologiczną.

Możliwe ulepszenia:
- zwiększenie liczby genów,
- dodanie większej liczby cech biologicznych,
- strojenie hiperparametrów,
- zastosowanie modeli SVM lub XGBoost.
""")
