import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([3, 6, 9, 12, 15])

# Model regresi linear
model = LinearRegression()
model.fit(X, Y)

# Prediksi untuk X = 6
X_pred = np.array([[6]])
Y_pred = model.predict(X_pred)

print(f"Prediksi Y untuk X = 6: {Y_pred[0]}")

#Hasilnya Y=18
