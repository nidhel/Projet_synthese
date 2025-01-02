import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/processed/data_processed.csv')
X = df.drop('species', axis=1)
y = df['species']

model = joblib.load('models/model.pkl')
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

with open('metrics.json', 'w') as f:
    f.write(f'{{"accuracy": {accuracy}}}')

print(f"✅ Évaluation terminée. Précision : {accuracy}")
