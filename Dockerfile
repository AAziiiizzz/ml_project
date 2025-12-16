# 1. Choix de l’image Python
FROM python:3.10-slim

# 2. Création du répertoire de travail
WORKDIR /app

# 3. Copier le requirements
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le projet dans le conteneur
COPY . .

# 6. Exposer le port FastAPI
EXPOSE 8000

# 7. Commande de lancement : uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
