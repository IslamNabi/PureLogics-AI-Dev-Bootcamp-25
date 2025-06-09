import pickle

def load_model():
    with open('app/models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model