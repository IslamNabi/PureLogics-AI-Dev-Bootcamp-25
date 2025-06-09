import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def create_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Ensure models directory exists
    os.makedirs('app/models', exist_ok=True)
    
    with open('app/models/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, iris.feature_names

if __name__ == '__main__':
    model, features = create_model()
    print("Model trained and saved successfully!")