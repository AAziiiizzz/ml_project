import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# ----------------------------
# Préprocessing
# ----------------------------

target_encoder = {'Y': 1, 'N': 0}
preprocessing_params = {}

def encode_categorical_features(df, is_train=True):
    df_encoded = df.copy()
    
    # Gender
    if 'Gender' in df_encoded.columns:
        mode_val = df_encoded['Gender'].mode()[0] if is_train else 'Male'
        df_encoded['Gender'] = df_encoded['Gender'].fillna(mode_val)
        df_encoded['Male'] = (df_encoded['Gender'] == 'Male').astype(int)
        df_encoded.drop('Gender', axis=1, inplace=True)

    # Married
    if 'Married' in df_encoded.columns:
        mode_val = df_encoded['Married'].mode()[0] if is_train else 'Yes'
        df_encoded['Married'] = df_encoded['Married'].fillna(mode_val)
        df_encoded['Married_Yes'] = (df_encoded['Married'] == 'Yes').astype(int)
        df_encoded.drop('Married', axis=1, inplace=True)

    # Dependents
    if 'Dependents' in df_encoded.columns:
        df_encoded['Dependents'] = df_encoded['Dependents'].fillna('0')
        rpl = {'0': 0, '1': 1, '2': 2, '3+': 3}
        df_encoded['Dependents'] = df_encoded['Dependents'].replace(rpl).astype(int)

    # Education
    if 'Education' in df_encoded.columns:
        df_encoded['Education_Graduate'] = (df_encoded['Education'] == 'Graduate').astype(int)
        df_encoded.drop('Education', axis=1, inplace=True)

    # Self_Employed
    if 'Self_Employed' in df_encoded.columns:
        mode_val = df_encoded['Self_Employed'].mode()[0] if is_train else 'No'
        df_encoded['Self_Employed'] = df_encoded['Self_Employed'].fillna(mode_val)
        df_encoded['Self_Employed_Yes'] = (df_encoded['Self_Employed'] == 'Yes').astype(int)
        df_encoded.drop('Self_Employed', axis=1, inplace=True)

    # Property_Area
    if 'Property_Area' in df_encoded.columns:
        property_area_dummies = pd.get_dummies(df_encoded['Property_Area'], prefix='Property_Area')
        df_encoded = pd.concat([df_encoded, property_area_dummies], axis=1)
        df_encoded.drop('Property_Area', axis=1, inplace=True)

    # Credit_History
    if 'Credit_History' in df_encoded.columns:
        mode_val = df_encoded['Credit_History'].mode()[0] if is_train else 1.0
        df_encoded['Credit_History'] = df_encoded['Credit_History'].fillna(mode_val)

    # LoanAmount
    if 'LoanAmount' in df_encoded.columns:
        mean_val = df_encoded['LoanAmount'].mean() if is_train else df_encoded['LoanAmount'].mean()
        df_encoded['LoanAmount'] = df_encoded['LoanAmount'].fillna(mean_val)

    # Drop Loan_Amount_Term
    if 'Loan_Amount_Term' in df_encoded.columns:
        df_encoded.drop('Loan_Amount_Term', axis=1, inplace=True)

    return df_encoded

def handle_missing_values(df, is_train=True):
    df_clean = df.copy()
    if is_train:
        preprocessing_params['mean_LoanAmount'] = df_clean['LoanAmount'].mean()
        preprocessing_params['mode_Credit_History'] = df_clean['Credit_History'].mode()[0]
    return df_clean

def preprocess_data(df, is_train=True):
    df_processed = encode_categorical_features(df, is_train)
    df_processed = handle_missing_values(df_processed, is_train)
    return df_processed

def prepare_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    train_processed = preprocess_data(train, is_train=True)
    test_processed = preprocess_data(test, is_train=False)

    X = train_processed.drop(['Loan_ID', 'Loan_Status'], axis=1, errors='ignore')
    y = train_processed['Loan_Status'].map(target_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_final_test = test_processed.drop(['Loan_ID'], axis=1, errors='ignore')

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_final_test': X_final_test,
        'test_ids': test_processed['Loan_ID'] if 'Loan_ID' in test_processed.columns else None
    }

# ----------------------------
# Modèle SVM
# ----------------------------

def train_model(X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    print("Modèle SVM entraîné avec succès")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy du modèle: {accuracy:.4f}")
    print(f"Prédictions correctes: {np.sum(y_test == y_pred)}")
    print(f"Prédictions incorrectes: {np.sum(y_test != y_pred)}")
    return {'accuracy': accuracy, 'predictions': y_pred, 'actual': y_test}

def predict(model, X_data):
    return model.predict(X_data)

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'preprocessing_params': preprocessing_params, 'target_encoder': target_encoder}, f)
    print(f"Modèle sauvegardé dans: {file_path}")

def load_model(file_path):
    with open(file_path, 'rb') as f:
        saved_data = pickle.load(f)
    global preprocessing_params, target_encoder
    preprocessing_params = saved_data['preprocessing_params']
    target_encoder = saved_data['target_encoder']
    print(f"Modèle chargé depuis: {file_path}")
    return saved_data['model']

def create_submission(predictions, test_ids, file_name='submission.csv'):
    submission = pd.DataFrame({
        'Loan_ID': test_ids,
        'Loan_Status': ['Y' if p == 1 else 'N' for p in predictions]
    })
    submission.to_csv(file_name, index=False)
    print(f"Fichier de soumission créé : {file_name}")
    return submission
