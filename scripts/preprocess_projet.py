import pandas as pd
import numpy as np
import os


# Vérifiez si le répertoire existe, sinon créez-le
os.makedirs('data/processed', exist_ok=True)
# Charger les données
df = pd.read_csv("data/raw/MetroPT3.csv", delimiter=",", decimal=".", index_col=0)
df.reset_index(drop=True, inplace=True)

# Convertir timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#print(df.dtypes)

################# A valider si prendre en consideration tout les cas ############################
# Identifier les doublons
#duplicated_rows = df[df.duplicated()]
#print(f"Nombre de doublons : {len(duplicated_rows)}")

# Détection des doublons dans les timestamps
#duplicated_timestamps = df[df['timestamp'].duplicated()]
#print(f"Nombre de doublons dans les timestamps : {len(duplicated_timestamps)}")

#####################
##### Étape 1 : #####
#####################
# Ajuster dabord les timestamp existant pour respecter le gap (saut) de 10 secondes.
def adjust_last_digit_of_seconds(timestamp):
    """Ajuste uniquement le dernier chiffre des secondes du timestamp."""
    seconds = timestamp.second
    last_digit = seconds % 10  # Extraire le dernier chiffre
    
    if 1 <= last_digit <= 5:
        # Arrondir à 0
        adjustment = -last_digit
    elif last_digit > 5:
        # Arrondir au multiple de 10 suivant
        adjustment = 10 - last_digit
    else:
        # Si le dernier chiffre est déjà 0
        adjustment = 0

    # Appliquer l'ajustement aux secondes
    return timestamp + pd.Timedelta(seconds=adjustment)

# Appliquer la correction à tous les timestamps
df['timestamp'] = df['timestamp'].apply(adjust_last_digit_of_seconds)



#####################
##### Étape 2 : #####
#####################

# Recalculer dynamiquement les intervalles après chaque correction
while True:
    # Calculer les intervalles
    time_diffs = df['timestamp'].diff().dt.total_seconds()

    # Identifier les intervalles problématiques (tout ce qui n'est pas 10 secondes)
    problematic_indices = time_diffs[time_diffs > 10].index

    if len(problematic_indices) == 0:
        print("Toutes les irrégularités ont été corrigées.")
        break

    # Cibler la première irrégularité détectée
    idx = problematic_indices[0]

    # Récupérer le timestamp de départ
    start_time = df.loc[idx - 1, 'timestamp']
    end_time = df.loc[idx, 'timestamp']

    # Calculer les timestamps intermédiaires pour combler l'intervalle
    new_times = pd.date_range(start=start_time, end=end_time, freq='10S', inclusive='right')

    # Créer un DataFrame avec les nouveaux timestamps
    missing_data = pd.DataFrame({'timestamp': new_times})

    # Ajouter les nouveaux timestamps dans le DataFrame
    df = pd.concat([df, missing_data]).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # Afficher l'état actuel
    print(f"Correction appliquée à l'intervalle autour de l'index {idx}.")
    print(df.loc[max(idx - 2, 0):min(idx + len(new_times) + 2, len(df) - 1), ['timestamp']])

# Vérification finale
time_diffs_corrected = df['timestamp'].diff().dt.total_seconds()
print("\nStatistiques des intervalles corrigés après correction finale :")
print(f"Min : {time_diffs_corrected.min()} secondes")
print(f"Max : {time_diffs_corrected.max()} secondes")
print(f"Mode (valeur typique) : {time_diffs_corrected.mode()[0]} secondes")
print(f"Nombre d'irrégularités restantes : {(time_diffs_corrected != 10).sum()}")    

# Ajouter les classes de panne
# Classe 0 : Pas de panne détectée
# Classe 1 : En pleine panne
# Classe 2 : Panne prévue dans moins de 30 minutes

# Liste des intervalles de pannes
pannes = [
    {'start': '2020-04-18 00:00', 'end': '2020-04-18 23:59'},
    {'start': '2020-05-29 23:30', 'end': '2020-05-30 06:00'},
    {'start': '2020-06-05 10:00', 'end': '2020-06-07 14:30'},
    {'start': '2020-07-15 14:30', 'end': '2020-07-15 19:00'},
         ]

# Convertir les timestamps des pannes en datetime
for panne in pannes:
    panne['start'] = pd.to_datetime(panne['start'])
    panne['end']   = pd.to_datetime(panne['end'])

# Ajouter une colonne 'panne' avec la valeur par défaut 0 (aucune panne détectée)
df['panne'] = 0


for panne in pannes:
    # Classe 1 : En pleine panne
    df.loc[(df['timestamp'] >= panne['start']) & (df['timestamp'] <= panne['end']), 'panne'] = 1
    # Classe 2 : Panne prévue dans moins de 30 minutes
    df.loc[(df['timestamp'] < panne['start']) & (df['timestamp'] >= panne['start'] - pd.Timedelta(minutes=30)), 'panne'] = 2

df.to_csv('data/processed/MetroPT3_corrected.csv', index=False)
print("✅ Données prétraitées et sauvegardées.")


