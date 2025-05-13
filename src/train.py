from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X, y, model_path='models/model.pkl'):
    """
    Trains and saves Logistic Regression model with TF-IDF vectors.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Save model to disk
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    
    return clf, X_val, y_val