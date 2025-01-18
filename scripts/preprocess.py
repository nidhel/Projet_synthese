import os
import pandas as pd

def preprocess():
    # Vérifiez si le répertoire existe, sinon créez-le
    os.makedirs('data/processed', exist_ok=True)
    
    df = pd.read_excel('data/raw/iris_Data.xls')
    df = df.dropna()
    df.to_csv('data/processed/data_processed.csv', index=False)
    print("✅ Données prétraitées et sauvegardées.")

if __name__ == "__main__":
    preprocess()
