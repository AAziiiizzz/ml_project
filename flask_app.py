from flask import Flask, render_template, request
import pandas as pd
from model_pipeline import preprocess_data, load_model, predict, prepare_data, save_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import time

app = Flask(__name__)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Approval_Experiment")

# Load initial model
model = load_model("SVM_loan_model.pkl")
trained_columns = model.feature_names_in_


# ----------------------------
# Home & Predict Form
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model, trained_columns

    if request.method == "POST":
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        # Convert numeric fields
        numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                        "Loan_Amount_Term", "Credit_History"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # Preprocess
        df_processed = preprocess_data(df, is_train=False)

        if "Loan_ID" in df_processed:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Missing columns
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        df_processed = df_processed[trained_columns]

        # Predict
        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return render_template("result.html", result=result)

    return render_template("index.html")


# ----------------------------
# Retrain (with MLflow logging)
# ----------------------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global model, trained_columns

    if request.method == "POST":
        C = float(request.form.get("C", 1.0))
        kernel = request.form.get("kernel", "rbf")
        gamma = request.form.get("gamma", "scale")

        # Load data
        data = prepare_data("data/train.csv", "data/test.csv")
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        # Old model evaluation
        old_pred = model.predict(X_test)
        old_acc = accuracy_score(y_test, old_pred)
        old_cm = confusion_matrix(y_test, old_pred).tolist()

        # New model
        new_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=kernel, C=C, gamma=gamma,
                        probability=True, class_weight='balanced',
                        random_state=42))
        ])

        # MLflow run
        with mlflow.start_run(run_name=f"Flask_Retrain_{int(time.time())}"):

            # Log params
            mlflow.log_param("C", C)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("gamma", gamma)

            new_model.fit(X_train, y_train)

            # Eval
            new_pred = new_model.predict(X_test)
            new_acc = accuracy_score(y_test, new_pred)
            new_cm = confusion_matrix(y_test, new_pred).tolist()

            # Log metrics
            mlflow.log_metric("old_accuracy", old_acc)
            mlflow.log_metric("new_accuracy", new_acc)

            # Log confusion matrices
            mlflow.log_dict({"old_confusion_matrix": old_cm}, "old_confusion_matrix.json")
            mlflow.log_dict({"new_confusion_matrix": new_cm}, "new_confusion_matrix.json")

            # Save model in MLflow
            mlflow.sklearn.log_model(new_model, "SVM_model")

        # Keep best
        if new_acc > old_acc:
            save_model(new_model, "SVM_loan_model.pkl")
            model = load_model("SVM_loan_model.pkl")
            trained_columns = model.feature_names_in_
            chosen = "new"
        else:
            save_model(new_model, "SVM_rejected.pkl")
            chosen = "old"

        comparison = {
            "old_model": {"accuracy": old_acc, "confusion_matrix": old_cm},
            "new_model": {"accuracy": new_acc, "confusion_matrix": new_cm,
                          "hyperparameters": {"C": C, "kernel": kernel, "gamma": gamma}},
            "better_model": chosen
        }

        return render_template("retrain.html", comparison=comparison)

    return render_template("retrain.html", comparison=None)


if __name__ == "__main__":
    app.run(debug=True)
