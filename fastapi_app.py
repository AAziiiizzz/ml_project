from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model_pipeline import preprocess_data, load_model, predict, prepare_data, save_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn

app = FastAPI()

# ---------------------------
# Load base model
# ---------------------------
model = load_model("SVM_loan_model.pkl")
trained_columns = model.feature_names_in_


# ---------------------------
# Request Body Schemas
# ---------------------------
class RetrainParams(BaseModel):
    C: float = 1.0
    kernel: str = "rbf"
    gamma: float | str = "scale"

class LoanData(BaseModel):
    Loan_ID: str
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float | None = None
    Credit_History: float
    Property_Area: str


# ---------------------------
# PREDICTION ENDPOINT
# ---------------------------
@app.post("/predict")
def predict_loan(data: LoanData):
    df = pd.DataFrame([data.dict()])
    df_processed = preprocess_data(df, is_train=False)

    if "Loan_ID" in df_processed:
        df_processed = df_processed.drop("Loan_ID", axis=1)

    for col in trained_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[trained_columns]

    pred = predict(model, df_processed)[0]
    result = "Approved" if pred == 1 else "Rejected"
    return {"Loan_ID": data.Loan_ID, "prediction": result}



# ---------------------------
# RETRAIN + MLFLOW ENDPOINT
# ---------------------------
@app.post("/retrain")
def retrain_model(params: RetrainParams):
    global model, trained_columns

    # Load data
    data = prepare_data("data/train.csv", "data/test.csv")
    X_train, X_test, y_train, y_test = (
        data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    )

    # OLD MODEL EVALUATION
    old_pred = model.predict(X_test)
    old_acc = accuracy_score(y_test, old_pred)
    old_cm = confusion_matrix(y_test, old_pred).tolist()

    # NEW MODEL
    new_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel=params.kernel,
            C=params.C,
            gamma=params.gamma,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # ---------------------------
    # ⬇️️ MLFLOW EXPERIMENT
    # ---------------------------
    mlflow.set_experiment("Loan_Approval_Experiment")

    with mlflow.start_run():
        # Log hyperparams
        mlflow.log_param("C", params.C)
        mlflow.log_param("kernel", params.kernel)
        mlflow.log_param("gamma", params.gamma)

        # Train model
        new_model.fit(X_train, y_train)

        # Evaluation
        new_pred = new_model.predict(X_test)
        new_acc = accuracy_score(y_test, new_pred)
        new_cm = confusion_matrix(y_test, new_pred).tolist()

        # Log metrics
        mlflow.log_metric("old_accuracy", old_acc)
        mlflow.log_metric("new_accuracy", new_acc)

        # Log confusion matrices
        mlflow.log_dict({"old_confusion_matrix": old_cm}, "old_confusion_matrix.json")
        mlflow.log_dict({"new_confusion_matrix": new_cm}, "new_confusion_matrix.json")

        # Log model
        mlflow.sklearn.log_model(new_model, "SVM_model")

        # Save model if better
        if new_acc > old_acc:
            save_model(new_model, "SVM_loan_model.pkl")
            model = load_model("SVM_loan_model.pkl")
            trained_columns = model.feature_names_in_
            chosen = "new"
        else:
            save_model(new_model, "SVM_rejected.pkl")
            chosen = "old"

    return {
        "comparison": {
            "old_model": {"accuracy": old_acc, "confusion_matrix": old_cm},
            "new_model": {
                "accuracy": new_acc,
                "confusion_matrix": new_cm,
                "hyperparameters": params.dict(),
            },
            "better_model": chosen,
        }
    }
