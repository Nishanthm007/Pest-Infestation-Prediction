# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/large_pest_infestation_dataset.csv")

# Encode categorical features
le_crop = LabelEncoder()
le_risk = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['Pest_Risk'] = le_risk.fit_transform(df['Pest_Risk'])

# Split data
X = df.drop("Pest_Risk", axis=1)
y = df["Pest_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model/pest_model.pkl")
joblib.dump(le_crop, "model/crop_encoder.pkl")
joblib.dump(le_risk, "model/risk_encoder.pkl")

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))
