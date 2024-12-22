# PINGUIN-CLASSIFICACIO

Aquest projecte implementa diversos classificadors de *machine *learning per a predir l'espècie de pingüins de l'Arxipèlag *Palmer basant-se en les seves característiques físiques.

## Estructura del Projecte

```
PINGUIN-CLASSIFICACIO/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── train_models.py
│   ├── data_preprocessing.py
│   ├── server.py
│   └── client.py
└── models/
    ├── logistic_regression.joblib
    ├── svm_classifier.joblib
    ├── decision_tree.joblib
    ├── knn_classifier.joblib
    └── preprocessor.joblib
```

## Instal·lació

1. Assegura't de tenir Poetry instal·lat.
2. Clona aquest repositori.
3. Executa poetry install al directori del projecte.

## Úso

1. Entrena els models:
```bash
poetry run python src/train_models.py
```

2. Inicia el servidor:
```bash
poetry run python src/server.py
```

3. En un altra terminal, executa el client per provar els models:
```bash
poetry run python src/client.py
```

## Models Implementats

- Regressió Logística
- Màquines de Vectors de Suport (SVM)
- Àrbres de Decisió
- KNN

## API Endpoints

- POST `/predict/logistic_regression`
- POST `/predict/svm_classifier`
- POST `/predict/decision_tree`
- POST `/predict/knn_classifier`

Cada endpoint espera un JSON amb els següents camps:
```json
{
    "bill_length_mm": float,
    "bill_depth_mm": float,
    "flipper_length_mm": float,
    "body_mass_g": float,
    "island": string,
    "sex": string
}
```