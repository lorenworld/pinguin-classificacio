import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

class PenguinDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.dict_vectorizer = DictVectorizer(sparse=False)
        self.label_mapping = None
        
    def load_and_clean_data(self):
        """Carregar i netejar el conjunt de dades de Palmer Penguins."""
        import seaborn as sns
        df = sns.load_dataset("penguins")
        return df.dropna()
    
    def prepare_features_and_target(self, df):
        """Preparar les característiques i la variable objectiu."""
        # Crear un mapeig de les espècies
        unique_species = sorted(df['species'].unique())
        self.label_mapping = {species: i for i, species in enumerate(unique_species)}
        
        # Convertir les espècies a valors numèrics
        y = df['species'].map(self.label_mapping)
        
        # Separar les característiques
        numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        categorical_features = ['island', 'sex']
        
        return df[numeric_features], df[categorical_features], y
    
    def transform_features(self, X_numeric, X_categorical, fit=False):
        """Transformar les característiques mitjançant StandardScaler i DictVectorizer."""
        # Normalitzar les característiques numèriques
        if fit:
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_numeric_scaled = self.scaler.transform(X_numeric)
        
        # Transformar les característiques categòriques
        categorical_dict = X_categorical.to_dict('records')
        if fit:
            X_categorical_encoded = self.dict_vectorizer.fit_transform(categorical_dict)
        else:
            X_categorical_encoded = self.dict_vectorizer.transform(categorical_dict)
        
        # Combinar les característiques
        return np.hstack([X_numeric_scaled, X_categorical_encoded])
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Preparar el conjunt de dades complet per a l'entrenament i la prova."""
        df = self.load_and_clean_data()
        X_numeric, X_categorical, y = self.prepare_features_and_target(df)
        
        # Separar les dades en conjunt d'entrenament i conjunt de prova
        X_numeric_train, X_numeric_test, X_categorical_train, X_categorical_test, y_train, y_test = \
            train_test_split(X_numeric, X_categorical, y, test_size=test_size, random_state=random_state)
        
        # Transformar les característiques
        X_train = self.transform_features(X_numeric_train, X_categorical_train, fit=True)
        X_test = self.transform_features(X_numeric_test, X_categorical_test, fit=False)
        
        return X_train, X_test, y_train, y_test

    def prepare_single_prediction(self, data_dict):
        """Preparar una sola instància per a la predicció."""
        numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        categorical_features = ['island', 'sex']
        
        X_numeric = pd.DataFrame([{k: float(data_dict[k]) for k in numeric_features}])
        X_categorical = pd.DataFrame([{k: data_dict[k] for k in categorical_features}])
        
        return self.transform_features(X_numeric, X_categorical, fit=False)
