# src/optimizer.py
import numpy as np
import pandas as pd
from orchestrator import evaluate_svm, evaluate_rf, evaluate_mlp

# ---
# Kernel RBF
# ---
def rbf_kernel(x1, x2, length_scale=1.0):
    """  
    Implementar kernel RBF  
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    squared_dist = np.sum((x1 - x2) ** 2)
    return np.exp(-squared_dist / (2 * length_scale ** 2))

# ---
# Ajuste del GP
# ---
def fit_gp(X, y, length_scale=1.0, noise=1e-6):
    """  
    Construir matriz K usando el kernel RBF  
    Resolver (K + noise*I)^(-1) y  
    Regresar los parámetros necesarios para predecir  
    """
    n = len(X)
    K = np.zeros((n, n))
    
    # Construir matriz de kernel
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf_kernel(X[i], X[j], length_scale)
    
    # Agregar ruido a la diagonal
    K_noisy = K + noise * np.eye(n)
    
    # Resolver sistema lineal: K_noisy * alpha = y
    try:
        alpha = np.linalg.solve(K_noisy, y)
    except np.linalg.LinAlgError:
        # Si hay problemas numéricos, usar pseudoinversa
        alpha = np.linalg.pinv(K_noisy) @ y
    
    return alpha, K, X

# ---
# Predicción del GP
# ---
def gp_predict(X_train, y_train, X_test, length_scale=1.0, noise=1e-6):
    """  
    Calcular media mu(x*) y varianza sigma^2(x*)  
    Para cada punto en X_test  
    """
    alpha, K, X_train_arr = fit_gp(X_train, y_train, length_scale, noise)
    
    mu = np.zeros(len(X_test))
    sigma = np.zeros(len(X_test))
    
    for i, x_star in enumerate(X_test):
        # Vector k(x*, X_train)
        k_star = np.array([rbf_kernel(x_star, x_train, length_scale) for x_train in X_train_arr])
        
        # Media predictiva
        mu[i] = k_star @ alpha
        
        # Varianza predictiva
        sigma_sq = rbf_kernel(x_star, x_star, length_scale) - k_star @ np.linalg.solve(K + noise * np.eye(len(K)), k_star)
        sigma[i] = max(0, sigma_sq)  # Asegurar no negatividad
    
    return mu, sigma

# ---
# Función de adquisición UCB
# ---
def acquisition_ucb(mu, sigma, kappa=2.0):
    """  
    UCB = mu + kappa * sigma
    """ 
    return mu + kappa * sigma

# ---
# Dominios discretos de búsqueda (según especificación)
# ---
def get_search_domain(model_name):
    """Retorna el dominio de búsqueda para cada modelo"""
    if model_name == "svm":
        # SVM: C ∈ {0.1, 1, 10, 100}, gamma ∈ {0.001, 0.01, 0.1, 1}
        domain = []
        for C in [0.1, 1, 10, 100]:
            for gamma in [0.001, 0.01, 0.1, 1]:
                domain.append({
                    'params': [C, gamma],
                    'numeric': [C, gamma]
                })
        return domain
    
    elif model_name == "rf":
        # RF: n_estimators ∈ {10, 20, 50, 100}, max_depth ∈ {2, 4, 6, 8}
        domain = []
        for n_est in [10, 20, 50, 100]:
            for depth in [2, 4, 6, 8]:
                domain.append({
                    'params': [n_est, depth],
                    'numeric': [n_est, depth]
                })
        return domain
    
    elif model_name == "mlp":
        # MLP: hidden_layer_sizes ∈ {(16,), (32,), (64,), (32,16)}, alpha ∈ {10^-4, 10^-3, 10^-2}
        domain = []
        hidden_configs = [(16,), (32,), (64,), (32, 16)]
        alphas = [1e-4, 1e-3, 1e-2]
        
        for hidden in hidden_configs:
            for alpha in alphas:
                # Representación numérica para el GP
                num_layers = len(hidden)
                total_neurons = sum(hidden)
                domain.append({
                    'params': (hidden, alpha),
                    'numeric': [num_layers, total_neurons, alpha]
                })
        return domain
    
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

# ---
# Función de evaluación para cada modelo
# ---
def evaluate_model(model_name, params):
    """Evalúa el modelo con los parámetros dados"""
    if model_name == "svm":
        C, gamma = params
        return evaluate_svm(C=C, gamma=gamma)
    elif model_name == "rf":
        n_estimators, max_depth = params
        return evaluate_rf(n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "mlp":
        hidden_layer_sizes, alpha = params
        return evaluate_mlp(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

# ---
# BO principal
# ---
def optimize_model(model_name, n_init=3, n_iter=10):
    """ 
    Optimización Bayesiana para seleccionar hiperparámetros
    """
    print(f"🚀 Iniciando BO para {model_name.upper()}...")
    
    # Obtener dominio de búsqueda
    domain = get_search_domain(model_name)
    print(f"Dominio de búsqueda: {len(domain)} puntos")
    
    # Historia de evaluaciones
    X_evaluated = []  # Representación numérica para GP
    y_evaluated = []  # Scores
    params_evaluated = []  # Parámetros originales
    
    # 1. Samplear puntos iniciales aleatorios
    initial_indices = np.random.choice(len(domain), size=min(n_init, len(domain)), replace=False)
    
    for idx in initial_indices:
        domain_point = domain[idx]
        params = domain_point['params']
        numeric_repr = domain_point['numeric']
        
        score = evaluate_model(model_name, params)
        
        X_evaluated.append(numeric_repr)
        y_evaluated.append(score)
        params_evaluated.append(params)
        
        print(f"  Punto inicial {len(X_evaluated)}: {params} -> R² = {score:.4f}")
    
    best_score = max(y_evaluated)
    best_params = params_evaluated[np.argmax(y_evaluated)]
    
    # 2. Ciclo iterativo de BO
    for iteration in range(n_iter):
        print(f"\n📊 Iteración {iteration + 1}/{n_iter}")
        
        # Ajustar GP con puntos observados
        X_array = np.array(X_evaluated)
        y_array = np.array(y_evaluated)
        
        # Crear array numérico del dominio para predicción
        domain_numeric = np.array([point['numeric'] for point in domain])
        
        # Predecir para todo el dominio
        mu, sigma = gp_predict(X_array, y_array, domain_numeric)
        
        # Calcular UCB
        ucb = acquisition_ucb(mu, sigma, kappa=2.0)
        
        # Seleccionar próximo punto (excluyendo ya evaluados)
        candidate_scores = []
        candidate_indices = []
        
        for i, point in enumerate(domain):
            # Verificar si ya fue evaluado
            already_evaluated = False
            for eval_params in params_evaluated:
                if model_name == "mlp":
                    # Para MLP, comparar tuplas
                    if point['params'][0] == eval_params[0] and abs(point['params'][1] - eval_params[1]) < 1e-6:
                        already_evaluated = True
                        break
                else:
                    # Para SVM y RF, comparar listas
                    if np.allclose(point['params'], eval_params, atol=1e-3):
                        already_evaluated = True
                        break
            
            if not already_evaluated:
                candidate_scores.append(ucb[i])
                candidate_indices.append(i)
        
        if not candidate_indices:
            print("  No hay más puntos por evaluar")
            break
        
        # Seleccionar mejor candidato por UCB
        best_candidate_idx = candidate_indices[np.argmax(candidate_scores)]
        next_domain_point = domain[best_candidate_idx]
        next_params = next_domain_point['params']
        next_numeric = next_domain_point['numeric']
        
        # Evaluar el punto seleccionado
        score = evaluate_model(model_name, next_params)
        
        # Actualizar historia
        X_evaluated.append(next_numeric)
        y_evaluated.append(score)
        params_evaluated.append(next_params)
        
        print(f"  Punto seleccionado: {next_params}")
        print(f"  UCB: {ucb[best_candidate_idx]:.4f}, R² real: {score:.4f}")
        
        # Actualizar mejor global
        if score > best_score:
            best_score = score
            best_params = next_params
            print(f"  🎯 NUEVO MEJOR! R² = {best_score:.4f}")
    
    print(f"\n✅ Optimización completada para {model_name.upper()}")
    print(f"Mejores parámetros: {best_params}")
    print(f"Mejor R² score: {best_score:.4f}")
    
    return best_params, best_score

# Script de prueba completo con los 3 modelos
if __name__ == "__main__":
    print("🧪 Probando Optimización Bayesiana con los 3 modelos...")
    
    # Probar con SVM
    print("\n" + "="*60)
    best_params_svm, best_score_svm = optimize_model("svm", n_init=2, n_iter=5)
    print(f"\n🎯 Resultado final SVM:")
    print(f"Parámetros: {best_params_svm}")
    print(f"R²: {best_score_svm:.4f}")
    
    # Probar con Random Forest
    print("\n" + "="*60)
    best_params_rf, best_score_rf = optimize_model("rf", n_init=2, n_iter=5)
    print(f"\n🎯 Resultado final Random Forest:")
    print(f"Parámetros: {best_params_rf}")
    print(f"R²: {best_score_rf:.4f}")
    
    # Probar con MLP
    print("\n" + "="*60)
    best_params_mlp, best_score_mlp = optimize_model("mlp", n_init=2, n_iter=5)
    print(f"\n🎯 Resultado final MLP:")
    print(f"Parámetros: {best_params_mlp}")
    print(f"R²: {best_score_mlp:.4f}")
    
    # Resumen comparativo
    print("\n" + "="*60)
    print("📊 RESUMEN COMPARATIVO DE OPTIMIZACIÓN BAYESIANA")
    print("="*60)
    print(f"SVM:      R² = {best_score_svm:.4f} - Parámetros: {best_params_svm}")
    print(f"RF:       R² = {best_score_rf:.4f} - Parámetros: {best_params_rf}")
    print(f"MLP:      R² = {best_score_mlp:.4f} - Parámetros: {best_params_mlp}")
    
    best_overall_score = max(best_score_svm, best_score_rf, best_score_mlp)
    best_model = "SVM" if best_score_svm == best_overall_score else "RF" if best_score_rf == best_overall_score else "MLP"
    print(f"\n🏆 MEJOR MODELO: {best_model} con R² = {best_overall_score:.4f}")