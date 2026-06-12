import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dane_projekt1.csv")

print(df.head())
print(df.info())


print(df.describe())


X = df.drop(columns=["Gene_ID", "Gene_Function"])
y = df["Gene_Function"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Klasy:", list(le.classes_))


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_s, y_train)


y_val_pred = model.predict(X_val_s)

print("\n=== VALIDATION ===")
print(classification_report(y_val, y_val_pred))


y_test_pred = model.predict(X_test_s)

print("\n=== TEST SET ===")
print(classification_report(y_test, y_test_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

import numpy as np

importances = model.feature_importances_
features = X.columns

sorted_idx = np.argsort(importances)[::-1]

print("\nFeature importance:")
for i in sorted_idx:
    print(features[i], ":", round(importances[i], 4))
