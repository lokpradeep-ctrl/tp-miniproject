import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title('House Price Prediction')
st.write('Enter the details below to predict the house price:')

# Input fields
area = st.number_input('Area (sqft)', min_value=500, max_value=10000, value=2000)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
age = st.number_input('Age of House (years)', min_value=0, max_value=100, value=10)

# Sample training data (same as in training script)
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 4100, 4200, 4300, 4400],
    'Bedrooms': [3, 4, 3, 5, 4, 5, 4, 5, 4, 5],
    'Age': [10, 15, 20, 5, 8, 12, 7, 3, 6, 2],
    'Price': [400, 500, 600, 650, 700, 750, 800, 850, 900, 950]
}
df = pd.DataFrame(data)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Train model (for demo; in production, load a pre-trained model)
model = LinearRegression()
model.fit(X, y)

if st.button('Predict Price'):
    input_data = np.array([[area, bedrooms, age]])
    prediction = model.predict(input_data)[0]
    st.success(f'Estimated House Price: ${prediction * 1000:,.2f}')
