import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os 
import sqlite3

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    load_model
)

DB_PATH = os.getenv("DB_PATH", "mlflow.db")  # mlflow.db pour local, variable pour CI

def init_db(path):
    """Crée la DB et les tables si elles n'existent pas."""
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Exemple de table pour MLflow (ajuster si besoin)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

# Initialisation de la DB avant tout
init_db(DB_PATH)


def main():
    # ================================
    # 🔥 Configuration MLflow
    # ================================
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    mlflow.set_experiment("Loan_Approval_Experiment")  # ✅ Même nom que Flask

    with mlflow.start_run(run_name="CLI_Training_Run"):

        # Chemins des fichiers
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'

        print("=== Pipeline de Prédiction de Prêts - Modèle SVM ===\n")

        # ================================
        # 1. Préparation des données
        # ================================
        print("1. Préparation des données...")
        data = prepare_data(train_path, test_path)

        mlflow.log_param("X_train_shape", str(data['X_train'].shape))
        mlflow.log_param("X_test_shape", str(data['X_test'].shape))
        mlflow.log_param("nb_features", data['X_train'].shape[1])
        mlflow.log_param("feature_names", list(data['X_train'].columns))

        # ================================
        # 2. Entraînement du modèle
        # ================================
        print("\n2. Entraînement du modèle SVM...")
        model = train_model(data['X_train'], data['y_train'])

        # Log hyperparams du SVM
        svc = model.named_steps["svc"]
        mlflow.log_param("C", svc.C)
        mlflow.log_param("kernel", svc.kernel)
        mlflow.log_param("gamma", svc.gamma)
        mlflow.log_param("class_weight", str(svc.class_weight))
        mlflow.log_param("random_state", svc.random_state)

        # ================================
        # 3. Évaluation du modèle
        # ================================
        print("\n3. Évaluation du modèle...")
        results = evaluate_model(model, data['X_test'], data['y_test'])

        # Log accuracy si disponible
        if "accuracy" in results:
            mlflow.log_metric("accuracy", results["accuracy"])
            print(f"✓ Accuracy: {results['accuracy']:.4f}")

        # Calculer toutes les métriques
        if "predictions" in results and "actual" in results:
            y_true = results["actual"]
            y_pred = results["predictions"]
            
            # Métriques de classification
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print(f"✓ Precision: {precision:.4f}")
            print(f"✓ Recall: {recall:.4f}")
            print(f"✓ F1-Score: {f1:.4f}")
            
            # ROC AUC
            try:
                y_proba = model.predict_proba(data['X_test'])[:, 1]
                roc_auc = roc_auc_score(y_true, y_proba)
                mlflow.log_metric("roc_auc", roc_auc)
                print(f"✓ ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"⚠ ROC AUC non calculé: {e}")

            # Matrice de confusion
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            mlflow.log_dict({
                "confusion_matrix": cm.tolist(),
                "TN": int(cm[0][0]),
                "FP": int(cm[0][1]),
                "FN": int(cm[1][0]),
                "TP": int(cm[1][1])
            }, "confusion_matrix.json")

        # ================================
        # 4. Prédictions sur test final
        # ================================
        print("\n4. Prédictions sur test final...")
        final_predictions = predict(model, data['X_final_test'])

        approved = int((final_predictions == 1).sum())
        rejected = int((final_predictions == 0).sum())
        total = len(final_predictions)
        
        mlflow.log_metric("approved", approved)
        mlflow.log_metric("rejected", rejected)
        mlflow.log_metric("total_predictions", total)
        mlflow.log_metric("approval_rate", approved / total)
        
        print(f"✓ Approved: {approved}")
        print(f"✓ Rejected: {rejected}")
        print(f"✓ Approval rate: {approved/total:.2%}")

        # ================================
        # 5. Sauvegarde du modèle
        # ================================
        print("\n5. Sauvegarde du modèle...")
        save_model(model, 'SVM_loan_model.pkl')

        # Enregistrer dans MLflow avec signature
        from mlflow.models.signature import infer_signature
        
        signature = infer_signature(data['X_train'], model.predict(data['X_train']))
        
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            signature=signature,
            input_example=data['X_train'].iloc[:5],
            registered_model_name="SVM_Loan_Model"
        )
        
        # Log aussi le fichier pkl
        mlflow.log_artifact('SVM_loan_model.pkl')
        
        print("✓ Modèle enregistré dans MLflow avec signature")

        # ================================
        # 6. Vérification
        # ================================
        print("\n6. Vérification du modèle chargé...")
        loaded_model = load_model('SVM_loan_model.pkl')
        test_predictions = predict(loaded_model, data['X_test'].head())

        mlflow.log_param("test_prediction_example", str(test_predictions))
        
        # Tags pour organiser
        mlflow.set_tag("training_type", "CLI")
        mlflow.set_tag("model_type", "SVM")
        mlflow.set_tag("environment", "development")
        mlflow.set_tag("data_source", "train.csv")

        # ================================
        # Résumé final
        # ================================
        #print("\n" + "="*60)
        #print("=== Pipeline terminé & loggué dans MLflow ===")
        #print("="*60)
        
        #run_id = mlflow.active_run().info.run_id
        #print(f"✅ MLflow Run ID: {run_id}")
        #print(f"🔗 View in UI: http://127.0.0.1:{5050}/#/experiments/1/runs/{run_id}")
        #print(f"\n💡 Pour voir les résultats, lancez: make mlflow")
        #print("="*60)

if __name__ == "__main__":
    main()