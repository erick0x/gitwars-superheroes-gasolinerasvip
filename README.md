# 🚀 Git Wars Superheroes - Equipo GasolinerasVIP

**👥 Integrantes:** 

- Erick José Fabián Sandoval
- Imanol Mendoza Saenz de Buruaga
- Saúl Isaías Bibiano Callejas
- Alejandro Iram Ramírez Nava

**🌐 API:** https://gasolinerasvip.onrender.com/docs  
**📚 Docs:** Revisar notebooks/nb_equipo_template.ipynb

## 📋 Elementos Completados

### ✅ Elemento 0 - Dataset
- Consumo de SuperHero API 
- Conversión automática de unidades
- Dataset final: `data/data.csv`

### ✅ Elemento 1 - Orquestador
- 3 modelos: SVM, Random Forest, MLP
- Evaluación con R² score
- `src/orchestrator.py`

### ✅ Elemento 2 - Optimización Bayesiana
- Implementación manual: GP + UCB
- Kernel RBF y ciclo iterativo
- `src/optimizer.py`

### ✅ Elemento 3 - Análisis Comparativo
- BO vs Random Search
- Gráficas y tablas comparativas
- `src/comparative_analysis.py`

### ✅ Elemento 4 - API + Docker + Render
- FastAPI con documentación automática
- Preprocesamiento en tiempo real
- Docker + Render deployment

## 🛠️ Comandos Locales

### Docker + Makefile

**Construir Docker:**
docker build -f deployments/Dockerfile -t superhero-api .

**Ejecutar:**
docker run -d -p 8000:8000 superhero-api

**Probar endpoints:**
curl http://localhost:8000/health
curl http://localhost:8000/info

$body = @{
    features = @{
        intelligence = 50
        strength = 80
        speed = 60
        durability = 70
        combat = 55
        height = "185 cm"
        weight = "90 kg"
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"

### Probar endpoints render

**Health**
Invoke-RestMethod -Uri "https://gasolinerasvip.onrender.com/health" -Method Get

**Info**
Invoke-RestMethod -Uri "https://gasolinerasvip.onrender.com/info" -Method Get

**Predict**
$body = @{features = @{intelligence=50; strength=80; speed=60; durability=70; combat=55; height="185 cm"; weight="90 kg"}} | ConvertTo-Json
Invoke-RestMethod -Uri "https://gasolinerasvip.onrender.com/predict" -Method Post -Body $body -ContentType "application/json"



