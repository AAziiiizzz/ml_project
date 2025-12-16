# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQUIREMENTS := requirements.txt

# Ports pour les serveurs
FLASK_PORT := 8001
FASTAPI_PORT := 8000
MLFLOW_PORT := 5050

# ----------------------------
# 1. Installation
# ----------------------------
install: $(VENV)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Environnement virtuel créé"

# ----------------------------
# 2. Préparer les données
# ----------------------------
prepare:
	$(PYTHON) main.py --prepare

# ----------------------------
# 3. Entraîner le modèle
# ----------------------------
train:
	$(PYTHON) main.py --train

# ----------------------------
# 4. Évaluer le modèle
# ----------------------------
evaluate:
	$(PYTHON) main.py --evaluate

# ----------------------------
# 5. Générer des prédictions / fichier de soumission
# ----------------------------
predict:
	$(PYTHON) main.py --predict

# ----------------------------
# 6. Lancer les tests rapides
# ----------------------------
test:
	$(PYTHON) test_pipeline.py

# ----------------------------
# 7. Vérification du code (CI)
# ----------------------------
lint:
	flake8 *.py
	black --check *.py
	bandit -r .

format:
	black *.py
	isort *.py

# ----------------------------
# 8. Nettoyage des fichiers générés
# ----------------------------
clean:
	rm -f *.pkl
	rm -f *.csv
	rm -rf _pycache_

# ----------------------------
# 9. Tout exécuter (pipeline complet)
# ----------------------------
all: install prepare train evaluate predict test

# ----------------------------
# 10. Surveiller les fichiers et relancer automatiquement
# ----------------------------
watch:
	@echo "=== Surveiller les fichiers avec watchmedo ==="
	watchmedo shell-command \
        --patterns=".py;.csv" \
        --recursive \
        --command='make all'

.PHONY: install prepare train evaluate predict test lint format clean all

security:
	bandit -r . -ll

# ----------------------------
# 11. Run Flask app
# ----------------------------
flask:
	@echo "=== Running Flask app on http://127.0.0.1:$(FLASK_PORT) ==="
	FLASK_APP=flask_app.py FLASK_ENV=development $(VENV)/bin/flask run --host=0.0.0.0 --port=$(FLASK_PORT)

# ----------------------------
# 12. Run FastAPI app
# ----------------------------
fastapi:
	@echo "=== Running FastAPI app on http://127.0.0.1:$(FASTAPI_PORT) ==="
	$(VENV)/bin/uvicorn fastapi_app:app --reload --host 0.0.0.0 --port=$(FASTAPI_PORT)

# ----------------------------
# 13. Run MLflow UI
# ----------------------------
mlflow:
	@echo "=== Starting MLflow UI on http://127.0.0.1:$(MLFLOW_PORT) ==="
	$(VENV)/bin/mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT)

mlflow-stop:
	@echo "Arrêt de MLflow..."
	@pkill -f "mlflow ui" || echo "MLflow n'était pas en cours d'exécution"
	@echo "✓ MLflow arrêté"

	
docker-build:
    docker build -t belhajyahia-aziz-4ds8-mlops .

docker-tag:
    docker tag belhajyahia-aziz-4ds8-mlops aziiiz0/belhajyahia-aziz-4ds8-mlops:latest

docker-push:
    docker push aziiiz0/belhajyahia-aziz-4ds8-mlops:latest

docker-run:
    docker run -p 8000:8000 aziiiz0/belhajyahia-aziz-4ds8-mlops
