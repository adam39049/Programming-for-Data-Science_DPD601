
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(df):
    # Encode target variable
    mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    df = df.copy()
    df["Target_encoded"] = df["Target"].map(mapping)

    # Separate features and target
    X = df.drop(columns=["Target", "Target_encoded"])
    y = df["Target_encoded"]

    print(f"Features shape: {X.shape}")
    print(f"Target shape  : {y.shape}")
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree trained.")
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n== {model_name} ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Dropout", "Enrolled", "Graduate"]
    ))
    return accuracy
