# src/train_best_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def train_and_save_best_model():
    """Entrena el mejor modelo RF con parámetros óptimos de BO y lo guarda"""
    print("Entrenando mejor modelo Random Forest...")
    
    # Cargar datos
    df = pd.read_csv('data/data.csv')
    X = df.drop('power', axis=1)
    y = df['power']
    
    # Parámetros óptimos encontrados por BO
    BEST_PARAMS = {
        'n_estimators': 10,
        'max_depth': 8
    }
    
    # Dividir datos (misma división que en orchestrator)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar características (igual que en orchestrator)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo con parámetros óptimos
    model = RandomForestRegressor(
        n_estimators=BEST_PARAMS['n_estimators'],
        max_depth=BEST_PARAMS['max_depth'],
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Modelo entrenado:")
    print(f"   - Parámetros: {BEST_PARAMS}")
    print(f"   - R² Train: {train_score:.4f}")
    print(f"   - R² Test:  {test_score:.4f}")
    
    # Crear directorio model si no existe
    os.makedirs('model', exist_ok=True)
    
    # Guardar modelo
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar scaler (importante para preprocesamiento en API)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Guardar nombres de características (para verificación)
    feature_names = list(X.columns)
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(" Modelo guardado en:")
    print("   - model/model.pkl")
    print("   - model/scaler.pkl")
    print("   - model/feature_names.pkl")

if __name__ == "__main__":
    train_and_save_best_model()