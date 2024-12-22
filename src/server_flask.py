from flask import Flask, request, jsonify 
import joblib
import os

app = Flask(__name__)

# Carregar els models i el preprocesador
models = {
    'logistic_regression': joblib.load('models/logistic_regression.joblib'),
    'svm_classifier': joblib.load('models/svm_classifier.joblib'),
    'decision_tree': joblib.load('models/decision_tree.joblib'),
    'knn_classifier': joblib.load('models/knn_classifier.joblib')
}
preprocessor = joblib.load('models/preprocessor.joblib')

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in models:
        return jsonify({'error': f'Model {model_name} no trobat'}), 404
    
    try:
        # Obtenir les dades de la petició
        data = request.json
        
        # Preparar les dades
        X = preprocessor.prepare_single_prediction(data)
        
        # Fer la predicció
        prediction = models[model_name].predict(X)
        
        # Convertir la predicció numèrica de nou al nom de l'espècie
        species = {v: k for k, v in preprocessor.label_mapping.items()}[prediction[0]]
        
        return jsonify({
            'prediction': species,
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
