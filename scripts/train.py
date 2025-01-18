import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import yaml
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ClassificationQualityMetric,
    ClassificationClassBalance,
)

# 📊 Étape 1 : Charger les paramètres
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)['train']

# 🔄 Étape 2 : Charger les données
df = pd.read_csv('data/processed/data_processed.csv')
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔗 Étape 3 : Connecter MLflow
mlflow.set_tracking_uri('https://cfd6-2607-fa49-f881-d700-ed46-8db3-fcc0-81e4.ngrok-free.app')
mlflow.set_experiment("RandomForest_Training")

with mlflow.start_run():
    # 🛠️ Étape 4 : Entraînement du Modèle
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 📊 Étape 5 : Calculer les métriques
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    # 🔗 Étape 6 : Logger les paramètres et métriques dans MLflow
    mlflow.log_param("n_estimators", params['n_estimators'])
    mlflow.log_param("max_depth", params['max_depth'])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 📝 Étape 7 : Sauvegarder le modèle localement et dans MLflow
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # 🔄 Étape 8 : Préparer les Données pour Evidently
    print("📊 Génération du rapport Evidently...")

    # Préparer les jeux de données de référence et actuels
    train_data = X_train.copy()
    train_data['target'] = y_train  # Colonne cible renommée pour Evidently
    train_data['prediction'] = model.predict(X_train)  # Prédictions pour le train

    test_data = X_test.copy()
    test_data['target'] = y_test  # Colonne cible renommée pour Evidently
    test_data['prediction'] = predictions  # Prédictions pour le test

    # Vérifier si les colonnes sont correctement alignées
    assert set(train_data.columns) == set(test_data.columns), "❌ Les colonnes ne sont pas alignées."

    # Générer le rapport Evidently
    report = Report(
        metrics=[
            DataDriftTable(),           # Détection des dérives des caractéristiques
            DatasetDriftMetric(),       # Analyse globale des dérives
            ClassificationQualityMetric(),  # Qualité de la classification
            ClassificationClassBalance(),  # Équilibre des classes
        ]
    )

    report.run(
        reference_data=train_data.reset_index(drop=True),
        current_data=test_data.reset_index(drop=True)
    )

    # Sauvegarder le rapport Evidently
    report_path = 'models/evidently_model_report.html'
    report.save_html(report_path)
    mlflow.log_artifact(report_path)

    # 📊 Étape 9 : Sauvegarder les métriques dans un fichier JSON
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    with open('metrics_train.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact('metrics_train.json')

    # ✅ Étape 10 : Afficher les Résultats
    print(f"✅ Modèle entraîné avec précision : {accuracy:.4f}")
    print(f"📊 Métriques sauvegardées dans metrics_train.json")
    print(f"📈 Rapport Evidently généré : {report_path}")
    print(f"🏃 View run at: https://cfd6-2607-fa49-f881-d700-ed46-8db3-fcc0-81e4.ngrok-free.app/#/experiments/0/runs/{mlflow.active_run().info.run_id}")
