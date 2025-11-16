# Makefile
.PHONY: build run stop status clean package train-model test-api test-predict info help

# Variables
IMAGE_NAME = superhero-api
CONTAINER_NAME = superhero-api-container
PORT = 8000

# Ayuda
help:
	@echo " Comandos disponibles para SuperHero API:"
	@echo ""
	@echo "  build         - Construir imagen Docker"
	@echo "  run           - Ejecutar contenedor en segundo plano" 
	@echo "  stop          - Detener y eliminar contenedor"
	@echo "  status        - Ver estado del contenedor"
	@echo "  clean         - Limpiar recursos Docker"
	@echo "  train-model   - Entrenar y guardar modelo óptimo"
	@echo "  test-api      - Probar endpoint /health"
	@echo "  test-predict  - Probar predicción con ejemplo"
	@echo "  info          - Ver información del modelo"
	@echo "  package       - Generar equipo_GasolinerasVIP.tar.gz"
	@echo "  all           - Ejecutar flujo completo de evaluación"
	@echo ""

# Construir imagen Docker
build:
	@echo " Construyendo imagen Docker..."
	docker build -f deployments/Dockerfile -t $(IMAGE_NAME) .

# Ejecutar contenedor
run:
	@echo " Iniciando contenedor..."
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo " Contenedor iniciado en http://localhost:$(PORT)"

# Ver estado del contenedor
status:
	@echo " Estado del contenedor:"
	docker ps -f name=$(CONTAINER_NAME)

# Detener contenedor
stop:
	@echo " Deteniendo contenedor..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo " Contenedor detenido y eliminado"

# Limpiar recursos Docker
clean:
	@echo " Limpiando recursos Docker..."
	docker system prune -f
	@echo " Recursos limpiados"

# Entrenar modelo
train-model:
	@echo " Entrenando modelo óptimo..."
	python src/train_best_model.py

# Probar API
test-api:
	@echo " Probando API..."
	curl -s http://localhost:$(PORT)/health | jq . || curl -s http://localhost:$(PORT)/health

# Probar predicción
test-predict:
	@echo " Probando predicción..."
	curl -X POST "http://localhost:$(PORT)/predict" \
		-H "Content-Type: application/json" \
		-d '{"features": {"intelligence": 75, "strength": 85, "speed": 70, "durability": 80, "combat": 65, "height": "6'\''1\"", "weight": "185 lb"}}' | jq . || \
	curl -X POST "http://localhost:$(PORT)/predict" \
		-H "Content-Type: application/json" \
		-d '{"features": {"intelligence": 75, "strength": 85, "speed": 70, "durability": 80, "combat": 65, "height": "6'\''1\"", "weight": "185 lb"}}'

# Información del modelo
info:
	@echo " Información del modelo..."
	curl -s http://localhost:$(PORT)/info | jq . || curl -s http://localhost:$(PORT)/info

# Generar paquete de entrega
package: train-model
	@echo " Generando paquete de entrega..."
	tar -czf equipo_GasolinerasVIP.tar.gz \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		--exclude='.vscode' \
		--exclude='*.png' \
		--exclude='*.jpg' \
		--exclude='*.ipynb_checkpoints' \
		--exclude='env' \
		--exclude='venv' \
		data/ src/ notebooks/ deployments/ api/ model/ Makefile README.md requirements.txt
	@echo " Paquete creado: equipo_GasolinerasVIP.tar.gz"
	@echo " Tamaño: $$(du -h equipo_GasolinerasVIP.tar.gz | cut -f1)"

# Flujo completo de evaluación (como lo hará el profesor)
all: stop build run
	@echo " Esperando que la API esté lista..."
	@sleep 5
	$(MAKE) test-api
	$(MAKE) info
	$(MAKE) test-predict
	@echo ""
	@echo " Evaluación completada - Todo funciona correctamente"
	@echo " API disponible en: http://localhost:$(PORT)"
	@echo " Documentación en: http://localhost:$(PORT)/docs"

# Comando para desarrollo
dev:
	@echo " Modo desarrollo - Ejecutando API directamente..."
	cd api && python main.py