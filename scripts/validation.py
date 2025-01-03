# validation.py

import pandas as pd
import streamlit as st

def validate_missing_data(df: pd.DataFrame) -> None:
    """
    Valide les données manquantes dans le DataFrame.
    Affiche des avertissements si les valeurs manquantes dépassent 10 %.
    
    Paramètres :
    - df (pd.DataFrame) : Le DataFrame à valider.
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100

    missing_columns = df.columns[df.isnull().any()].tolist()
    
    if not missing_columns:
        st.success("✅ Aucune valeur manquante détectée dans le fichier.")
    else:
        st.warning(f"⚠️ Colonnes avec valeurs manquantes : {', '.join(missing_columns)}")
        st.info(f"📉 Pourcentage total de valeurs manquantes : **{missing_percentage:.2f}%**")

    if missing_percentage > 10:
        st.warning(
            f"⚠️ **Attention : Plus de 10 % ({missing_percentage:.2f}%) des données sont manquantes.**\n"
            "Il est recommandé de vérifier les sources des données ou d'effectuer une analyse plus approfondie "
            "avant l'imputation."
        )
