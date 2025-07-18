import os
os.system('pip install tensorflow numpy')
import tensorflow as tf
import numpy as np, sys
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
app = Flask(__name__)
os.system('clear' if sys == 'linux' else 'cls')
# Generate synthetic training data (sunlight, water, soil_quality -> plant health)
np.random.seed(42)
num_samples = 500
sunlight = np.random.uniform(4, 12, num_samples)  # Hours of sunlight
water = np.random.uniform(0.5, 3.0, num_samples)  # Liters per day
soil_quality = np.random.uniform(0, 1, num_samples)  # Soil quality index (0 to 1)
# Define plant health score (simplified formula)
plant_health = 0.5 * sunlight + 1.2 * water + 3 * soil_quality + np.random.normal(0, 0.5, num_samples)
# Prepare input features and labels
x_train = np.column_stack((sunlight, water, soil_quality))
y_train = plant_health
x_train_scaled = scaler.fit_transform(x_train)
# Build model
model = tf.keras.Sequential([tf.keras.layers.Dense(1024, activation='relu', input_shape=(3,)), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(8, activation='relu'), tf.keras.layers.Dense(1, activation='linear')])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train_scaled, y_train, epochs=10, verbose=0)
os.system('clear' if sys == 'linux' else 'cls')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'sunlight' not in data or 'water' not in data or 'soil_quality' not in data:
        return jsonify({"error": "Not enough data inputted"}), 400
    sunlight = float(data['sunlight'])
    water = float(data['water'])
    soil = float(data['soil_quality']) / 10  # Normalize soil
    test_sample = np.array([[sunlight, water, soil]])
    test_sample_scaled = scaler.transform(test_sample)
    predicted_health = model.predict(test_sample_scaled)
    score = np.clip(predicted_health[0][0], 1, 10)
    status = "Poor" if score < 5 else "Moderate" if score < 7 else "Good"
    return jsonify({"usability_score": round(score, 2), "status": status})
if __name__ == "__main__":
    app.run(debug=False)