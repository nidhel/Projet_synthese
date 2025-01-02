from flask import Flask, request, jsonify
import joblib
import pandas as pd
import io
import os

# Crée l'application Flask
app = Flask(__name__)

# 📦 --- 1. Chargement des modèles LightGBM ---
# Charger tous les modèles d'imputation sauvegardés
MODEL_DIR = r"C:\AEC IA\Session 6\Projet de synthese\Projet\models"
imputation_models = {}

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith('.pkl'):
        col_name = model_file.replace('lightgbm_imputer_', '').replace('.pkl', '')
        model_path = os.path.join(MODEL_DIR, model_file)
        imputation_models[col_name] = joblib.load(model_path)

print(f"✅ {len(imputation_models)} modèles d'imputation chargés avec succès.")

# 📊 --- 2. Route de test ---
@app.route('/')
def home():
    return "L'API fonctionne et les modèles sont chargés !"

# 🛠️ --- 3. Route d'imputation ---
@app.route('/impute', methods=['POST'])
def impute():
    try:
        # Récupère le fichier CSV envoyé dans la requête
        file = request.files['file']

        if not file:
            return jsonify({"error": "Aucun fichier n'a été envoyé."}), 400

        # Lis le contenu du fichier
        content = file.read().decode('utf-8')
        data = pd.read_csv(io.StringIO(content))
        
        # Liste des colonnes imputables
        imputable_columns = list(imputation_models.keys())
        
        # Identifier les colonnes avec des valeurs manquantes
        missing_columns = [col for col in data.columns if col in imputable_columns and data[col].isnull().any()]
        
        if not missing_columns:
            return jsonify({"message": "Aucune colonne imputable n'a de valeurs manquantes."}), 200

        # Imputer les colonnes manquantes
        for col in missing_columns:
            model = imputation_models[col]
            missing_rows = data[data[col].isnull()]
            if not missing_rows.empty:
                predictions = model.predict(missing_rows.drop(columns=[col]))
                data.loc[data[col].isnull(), col] = predictions

        # Retourner le DataFrame avec les valeurs imputées
        return data.to_csv(index=False), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 --- 4. Lancement de l'API ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
