import numpy as np, tensorflow as tf, os, sys

# Generate synthetic training data (sunlight, water, soil_quality -> plant health)
np.random.seed(42)
num_samples = 500

sunlight = np.random.uniform(4, 12, num_samples)  # Hours of sunlight
water = np.random.uniform(0.5, 3.0, num_samples)  # Liters per day
soil_quality = np.random.uniform(0, 1, num_samples)  # Soil quality index (0 to 1)

# Define plant health score (simplified formula)
plant_health = 0.5 * sunlight + 1.2 * water + 3 * soil_quality + np.random.normal(0, 0.5, num_samples)

# Prepare input features and labels
X = np.column_stack((sunlight, water, soil_quality))
y = plant_health

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

os.system('cls' if sys== 'win' else 'clear')

# Compile the model
model.compile(optimizer='adamax', loss='mse')

# Train the model
model.fit(X, y, epochs=150, verbose=1)

import streamlit as st

# Streamlit UI
st.title("Plant Health Prediction")

# Input sliders for soil quality, water, and sunlight
soil_quality = st.slider("Soil Quality (on a scale from 1 to 10):", 1, 10, on_change=None)
water = st.slider("Water (in liters):", 0.5, 3.0, step=0.1, on_change=None)
sunlight = st.slider("Sunlight (in hours):", 1, 24, on_change=None)

# Function to predict plant health
def predict():
    soilentered = soil_quality / 10
    test_sample = np.array([[sunlight, water, soilentered]])  # user inputs
    predicted_health = model.predict(test_sample)

    if predicted_health[0][0] < 5:
        prediction_color = "red"
    elif predicted_health[0][0] < 7:
        prediction_color = "yellow"
    elif predicted_health[0][0] < 9:
        prediction_color = "green"

    if predicted_health[0][0] > 10:
        predicted_health[0][0] = 10

    st.markdown(f"<h3 style='color:{prediction_color};'>Predicted Plant Health: {predicted_health[0][0]:.2f}</h3>", unsafe_allow_html=True)

# Call the prediction function on button click
if st.button('Predict'):
    predict()

# Add quit button (optional, Streamlit usually doesn't need this)
if st.button('Quit'):
    st.stop()