import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Entrenamiento
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1)
model.fit(X, y)

# Predicción
pred = model.predict([[0.3, 0.6, 0.9]])
print("Predicción:", pred[0])