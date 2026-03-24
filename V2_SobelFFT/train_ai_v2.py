import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_v2():

    try:
        df = pd.read_csv('date_antrenare_v2.csv')
    except FileNotFoundError:
        print("Eroare: Nu am găsit 'date_antrenare_v2.csv'. Rulează mai întâi extractorul!")
        return

    X = df.drop('is_fake', axis=1)
    y = df['is_fake']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluare
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== REZULTATE MODEL V2 (CU FFT) ===")
    print(f"Noua Acuratețe: {acc * 100:.2f}%")

    importances = model.feature_importances_
    features = X.columns
    print("\nImportanța trăsăturilor:")
    for f, imp in zip(features, importances):
        print(f"- {f}: {imp * 100:.2f}%")

    joblib.dump(model, 'detector_ai_v2.pkl')
    print("\nModelul V2 a fost salvat.")


if __name__ == "__main__":
    train_v2()