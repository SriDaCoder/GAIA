import tensorflow as tf, numpy as np, random
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Generate training data
np.random.seed(random.randint(0, 1000))
num_samples = 1000
x_train = np.random.rand(num_samples, 4) * [14, 100, 14, 100]
y_train = np.clip(14 - (x_train[:, 1] / 10) - (x_train[:, 3] / 10) + (x_train[:, 2] / 2) - abs(x_train[:, 0] - 7), 1, 10)

# Normalize
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train_scaled, y_train, epochs=10, verbose=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = np.array([[data["pH"], data["turbidity"], data["dissolved_oxygen"], data["contaminants"]]])
    input_scaled = scaler.transform(input_data)
    return jsonify({"usability_score": round(np.clip(model.predict(input_scaled)[0][0], 1, 14), 2)})

if __name__ == "__main__":
    app.run(debug=True)