# src/train_classifier.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

IN_PATH = "artifacts/zoom_features_npz.npz"
MODEL_OUT = "artifacts/rf_keystroke_baseline.joblib"
os.makedirs("artifacts", exist_ok=True)

def load_data(path):
    d = np.load(path)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(IN_PATH)
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    # combine train+val for final training (optional). Here we'll train on train and validate separately.
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # evaluate
    y_tr_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    print("Train acc:", accuracy_score(y_train, y_tr_pred))
    print("Val   acc:", accuracy_score(y_val, y_val_pred))
    print("Test  acc:", accuracy_score(y_test, y_test_pred))

    print("\nClassification report (test):")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # optional: small confusion matrix print (summarized)
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix shape:", cm.shape)

    # save model
    joblib.dump(clf, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    main()
