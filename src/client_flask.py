import requests
import json

def test_prediction(data, model_name):
    """Realitza una petició de predicció al model especificat."""
    url = f'http://localhost:5000/predict/{model_name}'
    response = requests.post(url, json=data)
    print(f"\nPetició al {model_name}:")
    print(f"Dades d'entrada: {json.dumps(data, indent=2)}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")

def main():
    # Casos de prova
    test_cases = [
        {
            'bill_length_mm': 39.1,
            'bill_depth_mm': 18.7,
            'flipper_length_mm': 181.0,
            'body_mass_g': 3750.0,
            'island': 'Biscoe',
            'sex': 'Male'
        },
        {
            'bill_length_mm': 46.5,
            'bill_depth_mm': 17.9,
            'flipper_length_mm': 192.0,
            'body_mass_g': 3500.0,
            'island': 'Dream',
            'sex': 'Female'
        }
    ]
    
    # Models a provar
    models = ['logistic_regression', 'svm_classifier', 'decision_tree', 'knn_classifier']
    
    # Prova cada model amb cada cas de prova
    for data in test_cases:
        print("\n" + "="*50)
        print("Provant noves dades de pingüí")
        print("="*50)
        for model in models:
            test_prediction(data, model)

if __name__ == "__main__":
    main()
