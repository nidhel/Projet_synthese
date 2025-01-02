import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import joblib
import json

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)['train']

df = pd.read_csv('data/processed/data_processed.csv')
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir l'URI du serveur de suivi
#mlflow.set_tracking_uri('http://localhost:5000')  # ou l'URL de votre serveur MLflow
mlflow.set_tracking_uri('https://cfd6-2607-fa49-f881-d700-ed46-8db3-fcc0-81e4.ngrok-free.app')

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    # Enregistrer les paramètres et métriques avec MLflow
    mlflow.log_param("n_estimators", params['n_estimators'])
    mlflow.log_param("max_depth", params['max_depth'])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Sauvegarder le modèle localement
    joblib.dump(model, 'models/model.pkl')

    # Sauvegarder les métriques dans un fichier JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open('metrics_train.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✅ Modèle entraîné avec précision : {accuracy}")
    print(f"📊 Métriques sauvegardées dans metrics_train.json")
    print(f"🏃 View run at: http://localhost:5000/#/experiments/0/runs/{mlflow.active_run().info.run_id}")
