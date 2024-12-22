import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessament_dades import PenguinDataPreprocessor

def train_and_save_models():
    # Inicialitza el preprocesador
    preprocessor = PenguinDataPreprocessor()
    
    # Prepara les dades
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # Desa el preprocesador
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Defineix els models
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'svm_classifier': SVC(random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn_classifier': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Entrena i desa cada model
    for name, model in models.items():
        print(f"Entrenant {name}...")
        model.fit(X_train, y_train)
        
        # Avalua el model
        score = model.score(X_test, y_test)
        print(f"Precisió de {name}: {score:.4f}")
        
        # Desa el model
        joblib.dump(model, f'models/{name}.joblib')
        print(f"Model desat a models/{name}.joblib\n")

if __name__ == "__main__":
    train_and_save_models()
