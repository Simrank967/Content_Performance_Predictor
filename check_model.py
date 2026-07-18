import joblib

model = joblib.load("models/best_model.pkl")

print(type(model))
print(len(model.feature_names_))
print(model.feature_names_[-10:])