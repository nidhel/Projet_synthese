import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Charger les données
df = pd.read_csv("data/processed/MetroPT3_corrected.csv", delimiter=",", decimal=".", index_col=0)
df.reset_index(drop=True, inplace=True)

# Ajouter une colonne "_is_missing" pour chaque colonne existante
for col in df.columns:
    if col not in ['timestamp', 'panne']:      # Exclure 'timestamp' et 'panne'
        df[f"{col}_is_missing"] = df[col].isnull().astype(int)

# Calculer le nombre de `1` dans chaque colonne `_is_missing`
missing_counts = {col: df[col].sum() for col in df.columns if col.endswith('_is_missing')}

# Convertir les résultats en DataFrame pour un affichage clair
missing_counts_df = pd.DataFrame(list(missing_counts.items()), columns=['Colonne', 'Nombre de 1'])

# Enregistrer le dataframe corrigé dans un fichier CSV
df.to_csv("data/processed/MetroPT3_corrected_marked.csv", index=True)

# Charger les données
df = pd.read_csv("data/processed/MetroPT3_corrected_marked.csv", delimiter=",", decimal=".", index_col=0)
df.reset_index(drop=True, inplace=True)

# Convertir timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Colonnes marquant les valeurs manquantes (is_missing)
cols_is_missing = [col + '_is_missing' for col in ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
                                                   'Oil_temperature', 'Motor_current', 'COMP', 'DV_eletric', 
                                                   'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 
                                                   'Caudal_impulses']]

# Filtrer les colonnes sans "_is_missing"
columns_without_is_missing = [col for col in df.columns if not col.endswith('_is_missing')]

# Colonnes continues
cols_continuous = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']

# Colonnes binaires ou catégoriques
cols_categorical = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']

# Créer une copie du DataFrame pour l'imputation
df_imputed_6 = df.copy()

# Imputation des colonnes continues avec RandomForestRegressor
for target_column in cols_continuous:
    print(f"Imputation pour la colonne continue : {target_column}")

    # Diviser les données en lignes complètes et manquantes
    train_data = df[df[target_column].notna()]
    missing_data = df[df[target_column].isna()]

    if not train_data.empty and not missing_data.empty:
        # Préparer les données pour l'entraînement et la prédiction
        X_train = train_data[cols_continuous + cols_categorical].drop(columns=[target_column]).fillna(0)
        y_train = train_data[target_column]
        X_missing = missing_data[cols_continuous + cols_categorical].drop(columns=[target_column]).fillna(0)

        # Entraîner le modèle RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Prédire les valeurs manquantes
        predicted_values = model.predict(X_missing)

        # Remplir les valeurs manquantes dans le DataFrame
        df_imputed_6.loc[missing_data.index, target_column] = predicted_values
    else:
        print(f"Aucune donnée manquante pour la colonne continue : {target_column}.")

# Imputation des colonnes catégoriques avec RandomForestClassifier
for target_column in cols_categorical:
    print(f"Imputation pour la colonne catégorielle : {target_column}")

    # Diviser les données en lignes complètes et manquantes
    train_data = df[df[target_column].notna()]
    missing_data = df[df[target_column].isna()]

    if not train_data.empty and not missing_data.empty:
        # Préparer les données pour l'entraînement et la prédiction
        X_train = train_data[cols_continuous + cols_categorical].drop(columns=[target_column]).fillna(0)
        y_train = train_data[target_column]
        X_missing = missing_data[cols_continuous + cols_categorical].drop(columns=[target_column]).fillna(0)

        # Entraîner le modèle RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Prédire les valeurs manquantes
        predicted_values = model.predict(X_missing)

        # Remplir les valeurs manquantes dans le DataFrame
        df_imputed_6.loc[missing_data.index, target_column] = predicted_values
    else:
        print(f"Aucune donnée manquante pour la colonne catégorielle : {target_column}.")

# Enregistrer le dataframe corrigé dans un fichier CSV
df_imputed_6.to_csv("data/processed/MetroPT3_imputed_radom_forest.csv", index=True)