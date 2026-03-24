import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_model():
    print('Loading data...')
    df = pd.read_csv('training_data.csv')

    X = df.drop('is_fake', axis = 1)
    y = df['is_fake']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print('Training model...')
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Training Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfussion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'detector_ai_model.pkl')
    print("\nModel has been saved as 'detector_ai_model.pkl'")


if __name__ == "__main__":
    train_model()

