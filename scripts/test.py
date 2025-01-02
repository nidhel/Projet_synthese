import streamlit as st
import pandas as pd
import requests
import io

# 🌟 Titre de l'application
st.title('Imputation Automatique des Données avec LightGBM')

# 📤 Téléchargement du fichier CSV
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

# 🛠️ Traitement du fichier
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Aperçu du fichier téléchargé :**", df.head())

    # Identifier les colonnes avec des valeurs manquantes
    missing_columns = df.columns[df.isnull().any()].tolist()
    if not missing_columns:
        st.success("✅ Aucune valeur manquante détectée dans le fichier.")
    else:
        st.warning(f"⚠️ Colonnes avec valeurs manquantes : {', '.join(missing_columns)}")

    # 📨 Envoi des données à l'API pour imputation
    if st.button('🛠️ Imputer les données manquantes'):
        csv_data = df.to_csv(index=False)
        response = requests.post(
            "http://localhost:5000/impute",
            files={"file": ("file.csv", io.StringIO(csv_data))}
        )

        if response.status_code == 200:
            imputed_df = pd.read_csv(io.StringIO(response.text))
            st.success("✅ Données imputées avec succès !")
            st.write("📊 **Aperçu des données imputées :**", imputed_df.head())
        else:
            st.error(f"❌ Erreur API : {response.text}")
