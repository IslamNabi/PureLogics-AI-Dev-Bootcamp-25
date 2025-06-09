from flask import jsonify, request, render_template
import numpy as np
from .models.model import load_model

def init_routes(app):
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        try:
            features = np.array(data['features']).reshape(1, -1)
            model = load_model()
            prediction = model.predict(features).tolist()
            proba = model.predict_proba(features).tolist()
            
            return jsonify({
                'prediction': prediction,
                'probability': proba
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500