# src/comparative_analysis.py (VERSIÓN CORREGIDA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optimizer import optimize_model
from random_search import random_search

# Configurar estilo para gráficas (SIN EMOJIS)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_comparative_analysis(n_iter=10):
    """
    Ejecuta análisis comparativo completo BO vs Random Search
    """
    print("INICIANDO ANALISIS COMPARATIVO BO vs RANDOM SEARCH")
    print("=" * 60)
    
    results = {}
    
    # Para cada modelo, ejecutar BO y Random Search
    for model_name in ["svm", "rf", "mlp"]:
        print(f"\nAnalizando modelo: {model_name.upper()}")
        print("-" * 40)
        
        # Optimización Bayesiana
        print("Ejecutando Optimización Bayesiana...")
        bo_params, bo_score = optimize_model(model_name, n_init=3, n_iter=n_iter)
        
        # Random Search  
        print("\nEjecutando Random Search...")
        rs_params, rs_score, rs_history = random_search(model_name, n_iter=n_iter)
        
        # Almacenar resultados
        results[model_name] = {
            'bo': {'params': bo_params, 'score': bo_score},
            'random_search': {'params': rs_params, 'score': rs_score, 'history': rs_history},
            'improvement': bo_score - rs_score
        }
        
        print(f"\nComparacion {model_name.upper()}:")
        print(f"   BO:           R² = {bo_score:.4f}")
        print(f"   Random Search: R² = {rs_score:.4f}")
        print(f"   Mejora BO:    +{results[model_name]['improvement']:.4f}")
    
    return results

def create_comparison_table(results):
    """
    Crea tabla comparativa de resultados
    """
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA: BO vs RANDOM SEARCH")
    print("=" * 80)
    
    table_data = []
    for model_name, result in results.items():
        table_data.append({
            'Modelo': model_name.upper(),
            'BO Parámetros': str(result['bo']['params']),
            'BO R²': f"{result['bo']['score']:.4f}",
            'RS Parámetros': str(result['random_search']['params']),
            'RS R²': f"{result['random_search']['score']:.4f}",
            'Mejora': f"+{result['improvement']:.4f}"
        })
    
    df_comparison = pd.DataFrame(table_data)
    print(df_comparison.to_string(index=False))
    
    return df_comparison

def plot_convergence_analysis(results):
    """
    Crea gráficas de convergencia y análisis comparativo
    """
    # Tamaño más pequeño y sin emojis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Analisis Comparativo: BO vs Random Search', fontsize=14, fontweight='bold')
    
    # 1. Gráfica de barras comparativa
    models = list(results.keys())
    bo_scores = [results[model]['bo']['score'] for model in models]
    rs_scores = [results[model]['random_search']['score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, bo_scores, width, label='BO', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, rs_scores, width, label='Random Search', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Modelo')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Comparacion de Rendimiento Final')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (bo, rs) in enumerate(zip(bo_scores, rs_scores)):
        axes[0, 0].text(i - width/2, bo + 0.01, f'{bo:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 0].text(i + width/2, rs + 0.01, f'{rs:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Gráfica de mejora
    improvements = [results[model]['improvement'] for model in models]
    colors = ['green' if imp >= 0 else 'red' for imp in improvements]
    
    bars = axes[0, 1].bar(models, improvements, color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('Modelo')
    axes[0, 1].set_ylabel('Mejora (R²)')
    axes[0, 1].set_title('Mejora de BO sobre Random Search')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'+{imp:.4f}' if imp >= 0 else f'{imp:.4f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # 3. Gráfica de rendimiento por modelo
    x_model = np.arange(len(models))
    axes[1, 0].plot(x_model, bo_scores, marker='o', linewidth=2, markersize=8, label='BO')
    axes[1, 0].plot(x_model, rs_scores, marker='s', linewidth=2, markersize=8, label='Random Search')
    
    axes[1, 0].set_xlabel('Modelo')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Rendimiento por Modelo y Metodo')
    axes[1, 0].set_xticks(x_model)
    axes[1, 0].set_xticklabels([m.upper() for m in models])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Resumen estadístico (sin emojis)
    axes[1, 1].axis('off')
    summary_text = (
        "RESUMEN ESTADISTICO\n\n"
        f"Mejor modelo BO: {models[np.argmax(bo_scores)].upper()} (R² = {max(bo_scores):.4f})\n"
        f"Mejor modelo RS: {models[np.argmax(rs_scores)].upper()} (R² = {max(rs_scores):.4f})\n"
        f"Mejora promedio: {np.mean(improvements):.4f}\n"
        f"BO gana en: {sum(1 for imp in improvements if imp > 0)}/3 modelos\n"
        f"Mayor mejora: {max(improvements):.4f} ({models[np.argmax(improvements)].upper()})"
    )
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('bo_vs_random_search_comparison.png', dpi=150, bbox_inches='tight')  # DPI más bajo
    plt.show()

def generate_interpretation(results):
    """
    Genera análisis interpretativo de los resultados
    """
    print("\n" + "=" * 80)
    print("ANALISIS INTERPRETATIVO")
    print("=" * 80)
    
    # Análisis por modelo
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"   - BO: {result['bo']['params']} -> R² = {result['bo']['score']:.4f}")
        print(f"   - RS: {result['random_search']['params']} -> R² = {result['random_search']['score']:.4f}")
        print(f"   - Mejora: {result['improvement']:.4f}")
    
    # Análisis general
    improvements = [results[model]['improvement'] for model in results.keys()]
    bo_wins = sum(1 for imp in improvements if imp > 0)
    
    print(f"\nCONCLUSIONES GENERALES:")
    print(f"   1. BO supera a Random Search en {bo_wins}/3 modelos")
    print(f"   2. Mejora promedio de BO: {np.mean(improvements):.4f} puntos R²")
    print(f"   3. Mayor mejora en: {list(results.keys())[np.argmax(improvements)].upper()}")
    
    print(f"\nEXPLICACION TECNICA:")
    print("   - BO converge mas rapido porque:")
    print("     * Usa Gaussian Process para modelar la funcion objetivo")
    print("     * UCB balancea exploracion (alta incertidumbre) y explotacion (alta media)")
    print("     * Aprende de evaluaciones previas para guiar la busqueda")
    print("   - Random Search explora aleatoriamente sin aprendizaje")
    
    print(f"\nOBSERVACIONES POR MODELO:")
    best_bo_model = max(results.keys(), key=lambda x: results[x]['bo']['score'])
    best_rs_model = max(results.keys(), key=lambda x: results[x]['random_search']['score'])
    
    print(f"   - Mejor modelo con BO: {best_bo_model.upper()}")
    print(f"   - Mejor modelo con RS: {best_rs_model.upper()}")
    print(f"   - Modelo mas estable: RF (generalmente buen rendimiento en ambos metodos)")

def main():
    """
    Función principal del análisis comparativo
    """
    print("INICIANDO ELEMENTO 3 - ANALISIS COMPARATIVO")
    print("BO vs RANDOM SEARCH")
    print("=" * 60)
    
    # Ejecutar análisis comparativo
    results = run_comparative_analysis(n_iter=8)
    
    # Generar tabla comparativa
    df_table = create_comparison_table(results)
    
    # Crear gráficas
    plot_convergence_analysis(results)
    
    # Generar análisis interpretativo
    generate_interpretation(results)
    
    print(f"\nANALISIS COMPLETADO")
    print("Resultados guardados en:")
    print("   - bo_vs_random_search_comparison.png (graficas)")
    print("   - Memoria tecnica en consola")

if __name__ == "__main__":
    main()