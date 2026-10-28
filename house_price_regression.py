import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: House prices
# Features: Area (sqft), Bedrooms, Age (years)
# Target: Price (in $1000s)
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 4100, 4200, 4300, 4400],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 4, 5, 4, 5],
    'Age': [10, 15, 20, 5, 8, 12, 7, 3, 6, 2],
    'Price': [400, 500, 600, 650, 700, 750, 800, 850, 900, 950]
}
df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print('Test Predictions:', y_pred)
print('Mean Squared Error:', mse)
print('Model Coefficients:', model.coef_)
print('Model Intercept:', model.intercept_)
