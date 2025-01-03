import pandas as pd

# Charger les datasets
rf_imputed = pd.read_csv("data/processed/MetroPT3_imputed_radom_forest.csv")  # Imputé avec Random Forest
xgb_imputed = pd.read_csv("data/processed/MetroPT3_imputed_xgboost.csv")  # Imputé avec XGBoost


# Conserver la structure initiale
final_df = rf_imputed.copy()

# Remplacer la colonne Oil_temperature par celle imputée via XGBoost
final_df['Oil_temperature'] = xgb_imputed['Oil_temperature']

# Sauvegarder le dataset final
final_df.to_csv("data/processed/MetroPT3_imputed_final.csv", index=False)

print("Le dataset final a été enregistré sous ../Datasources/MetroPT3_imputed_final.csv")