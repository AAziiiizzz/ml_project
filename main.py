# main.py

import warnings
warnings.filterwarnings('ignore')

# Importer toutes les fonctions et variables globales depuis model_pipeline.py
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    load_model
)

def main():
    # Chemins des fichiers de données
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    print("=== Pipeline de Prédiction de Prêts - Modèle SVM ===\n")

    # Étape 1: Préparation des données
    print("1. Préparation des données...")
    data = prepare_data(train_path, test_path)
    print(f"Shape X_train: {data['X_train'].shape}")
    print(f"Shape X_test: {data['X_test'].shape}")
    print(f"Shape X_final_test: {data['X_final_test'].shape}")
    print(f"Features utilisées: {list(data['X_train'].columns)}\n")

    # Étape 2: Entraînement du modèle
    print("2. Entraînement du modèle SVM...")
    model = train_model(data['X_train'], data['y_train'])
    print()

    # Étape 3: Évaluation du modèle
    print("3. Évaluation du modèle...")
    results = evaluate_model(model, data['X_test'], data['y_test'])
    print()

    # Étape 4: Prédictions sur les données de test finales
    print("4. Prédictions sur les données de test...")
    final_predictions = predict(model, data['X_final_test'])
    print(f"Nombre de prédictions: {len(final_predictions)}")
    print(f"Prêts approuvés: {sum(final_predictions == 1)}")
    print(f"Prêts refusés: {sum(final_predictions == 0)}\n")

    # Étape 5: Sauvegarde du modèle
    print("5. Sauvegarde du modèle...")
    save_model(model, 'naive_bayes_loan_model.pkl')
    print()

    # Étape 6: Chargement du modèle et vérification
    print("6. Test de chargement du modèle...")
    loaded_model = load_model('naive_bayes_loan_model.pkl')
    test_predictions = predict(loaded_model, data['X_test'].head())
    print(f"Prédictions de test avec modèle chargé: {test_predictions}\n")

    print("=== Pipeline terminé avec succès ===")

if __name__ == "__main__":
    main()