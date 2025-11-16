# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List, Dict
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS PYDANTIC
# =============================================================================

class FeaturesRaw(BaseModel):
    intelligence: Union[float, int] = 0
    strength: Union[float, int] = 0
    speed: Union[float, int] = 0
    durability: Union[float, int] = 0
    combat: Union[float, int] = 0
    height: Union[str, float, int] = "0'0"  # Puede ser "3'2"", "6'8", "180 cm", o 180.0
    weight: Union[str, float, int] = "0 kg"  # Puede ser "180 lb", "80 kg", o 80.0

class PredictionRequest(BaseModel):
    features: FeaturesRaw

class PredictionResponse(BaseModel):
    prediction: float
    processed_features: Dict[str, float]
    timestamp: str

class InfoResponse(BaseModel):
    team_name: str
    model_type: str
    optimal_parameters: Dict
    preprocessing: List[str]
    api_version: str
    endpoints: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    scaler_loaded: bool

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO MEJORADAS
# =============================================================================

def convert_to_cm(height_input: Union[str, float, int]) -> float:
    """
    Convierte altura a centímetros - MEJORADA para pulgadas
    """
    if height_input is None:
        raise ValueError("La altura no puede ser None")
    
    # Si ya es numérico, asumir que está en cm
    if isinstance(height_input, (int, float)):
        return float(height_input)
    
    height_str = str(height_input).strip()
    
    # Caso: "-" o vacío
    if not height_str or height_str == "-":
        raise ValueError("Formato de altura no válido")
    
    # Si ya está en cm
    if "cm" in height_str.lower():
        try:
            return float(height_str.lower().replace("cm", "").strip())
        except ValueError:
            pass
    
    # Si está en pies y pulgadas (formato: "6'8"")
    if "'" in height_str:
        try:
            parts = height_str.split("'")
            feet = float(parts[0])
            inches_str = parts[1].replace('"', '').strip()
            inches = float(inches_str) if inches_str else 0
            cm = (feet * 30.48) + (inches * 2.54)
            return round(cm, 2)
        except (ValueError, IndexError):
            pass
    
    # NUEVO: Si está solo en pulgadas (formato: "72" o "72in")
    if '"' in height_str or "in" in height_str.lower():
        try:
            inches_str = height_str.replace('"', '').replace('in', '').replace('IN', '').strip()
            inches = float(inches_str)
            cm = inches * 2.54
            return round(cm, 2)
        except ValueError:
            pass
    
    # Intentar convertir directamente
    try:
        return float(height_str)
    except ValueError:
        raise ValueError(f"No se pudo convertir altura: {height_input}")

def convert_to_kg(weight_input: Union[str, float, int]) -> float:
    """
    Convierte peso a kilogramos
    """
    if weight_input is None:
        raise ValueError("El peso no puede ser None")
    
    # Si ya es numérico, asumir que está en kg
    if isinstance(weight_input, (int, float)):
        return float(weight_input)
    
    weight_str = str(weight_input).strip()
    
    # Caso: "-" o vacío
    if not weight_str or weight_str == "-":
        raise ValueError("Formato de peso no válido")
    
    # Si ya está en kg
    if "kg" in weight_str.lower():
        try:
            return float(weight_str.lower().replace("kg", "").strip())
        except ValueError:
            pass
    
    # Si está en libras
    if "lb" in weight_str.lower():
        try:
            lbs = float(weight_str.lower().replace("lb", "").strip())
            kg = lbs * 0.453592
            return round(kg, 2)
        except ValueError:
            pass
    
    # Intentar convertir directamente
    try:
        return float(weight_str)
    except ValueError:
        raise ValueError(f"No se pudo convertir peso: {weight_input}")

# =============================================================================
# CONFIGURACIÓN FASTAPI
# =============================================================================

app = FastAPI(
    title="SuperHero Power Predictor API",
    description="""
    API para predecir el poder de superhéroes
    
    Características:
    - Modelo Random Forest optimizado con Bayesian Optimization
    - Preprocesamiento automático de unidades (pies → cm, libras → kg)
    - Listo para deployment en Render
    
    Equipo: GasolinerasVIP
    """,
    version="2.1.0",
    contact={
        "name": "Equipo GasolinerasVIP",
        "url": "https://github.com/erick0x/gitwars-superheroes-lab10",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =============================================================================
# CARGA DE MODELOS
# =============================================================================

def load_model_resources():
    """Carga el modelo, scaler y recursos necesarios"""
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Modelo cargado correctamente")
        
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("✅ Scaler cargado correctamente")
        
        with open('model/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        logger.info("✅ Nombres de características cargados")
        
        return model, scaler, feature_names
        
    except Exception as e:
        logger.error(f"❌ Error cargando recursos: {e}")
        raise RuntimeError(f"No se pudieron cargar los recursos del modelo: {e}")

# Cargar recursos al inicio
try:
    model, scaler, feature_names = load_model_resources()
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Error inicializando la app: {e}")
    model, scaler, feature_names = None, None, None
    MODEL_LOADED = False

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Página de inicio redirige a docs"""
    return {
        "message": "Bienvenido a SuperHero Power Predictor API",
        "team": "GasolinerasVIP",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificación de salud del servicio y recursos"""
    return HealthResponse(
        status="ok" if MODEL_LOADED else "error",
        timestamp=datetime.now().isoformat(),
        model_loaded=MODEL_LOADED,
        scaler_loaded=MODEL_LOADED
    )

@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Información completa del equipo, modelo y preprocesamiento"""
    return InfoResponse(
        team_name="GasolinerasVIP",
        model_type="Random Forest Regressor",
        optimal_parameters={
            "n_estimators": 10,
            "max_depth": 8,
            "random_state": 42
        },
        preprocessing=[
            "Conversión automática de unidades: pies/pulgadas/libras a cm/kg",
            "StandardScaler para normalización de características",
            "Manejo de valores faltantes con imputación por media"
        ],
        api_version="2.1.0",
        endpoints=[
            "GET /health - Verificación de salud",
            "GET /info - Información del modelo", 
            "POST /predict - Predicción de poder",
            "GET /docs - Documentación interactiva"
        ]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_power(request: PredictionRequest):
    """
    Predice el poder de un superhéroe basado en sus características
    
    Características de entrada:
    - intelligence, strength, speed, durability, combat: Valores entre 0-100
    - height: Puede ser "3'2"", "6'8"", "72in", "180 cm", o 180.0
    - weight: Puede ser "980 lb", "80 kg", o 80.0
    
    Ejemplo de entrada en cm:
    ```json
    {
        "features": {
            "intelligence": 75,
            "strength": 85, 
            "speed": 70,
            "durability": 80,
            "combat": 65,
            "height": "185",
            "weight": "85 kg"
        }
    }
    ```
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="Servicio no disponible - Modelo no cargado"
        )
    
    try:
        logger.info(f"📥 Recibida solicitud de predicción")
        
        # 1. VALIDACIÓN DE RANGOS (0-100)
        stats = {
            'intelligence': request.features.intelligence,
            'strength': request.features.strength,
            'speed': request.features.speed, 
            'durability': request.features.durability,
            'combat': request.features.combat
        }
        
        invalid_stats = []
        for stat_name, stat_value in stats.items():
            if stat_value > 100:
                invalid_stats.append(f"{stat_name}: {stat_value} (máximo permitido: 100)")
        
        if invalid_stats:
            error_msg = "Las siguientes características exceden el valor máximo de 100:\n" + "\n".join(invalid_stats)
            raise ValueError(error_msg)
        
        # 2. CONVERSIÓN DE UNIDADES
        height_cm = convert_to_cm(request.features.height)
        weight_kg = convert_to_kg(request.features.weight)
        
        logger.info(f"🔧 Unidades convertidas: {request.features.height} -> {height_cm} cm, "
                   f"{request.features.weight} -> {weight_kg} kg")
        
        # 3. PREPARACIÓN DE CARACTERÍSTICAS
        input_features = np.array([[
            request.features.intelligence,
            request.features.strength,
            request.features.speed,
            request.features.durability, 
            request.features.combat,
            height_cm,
            weight_kg
        ]])
        
        # 4. APLICAR ESCALADO
        features_scaled = scaler.transform(input_features)
        
        # 5. PREDICCIÓN
        prediction = model.predict(features_scaled)
        prediction_value = float(np.clip(prediction[0], 0, 100))
        
        logger.info(f"🎯 Predicción generada: {prediction_value:.2f}")
        
        return PredictionResponse(
            prediction=round(prediction_value, 2),
            processed_features={
                "intelligence": request.features.intelligence,
                "strength": request.features.strength,
                "speed": request.features.speed,
                "durability": request.features.durability,
                "combat": request.features.combat,
                "height_cm": height_cm,
                "weight_kg": weight_kg
            },
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error interno: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando SuperHero Power Predictor API...")
    if MODEL_LOADED:
        logger.info("✅ API lista para recibir solicitudes")
    else:
        logger.error("❌ API iniciada pero modelo no cargado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000,
        log_level="info"
    )