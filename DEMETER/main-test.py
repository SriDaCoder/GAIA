import os

os.system('pip install tk tensorflow numpy')
import tensorflow as tf
import numpy as np, sys, tkinter as tk

os.system('clear' if sys == 'linux' else 'cls')

# Create a GUI window
root = tk.Tk()
root.title("Plant Health Prediction")
root.attributes('-topmost', True)
root.configure(bg="#000000")

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
    tf.keras.layers.Dense(1)
])

os.system('clear' if sys == 'linux' else 'cls')

tk.Label(root, text="Soil Quality (on a scale from 1-10):", font=("Helvetica", 12), bg="#000000", fg="white").grid(row=0, column=0)
soil_entry = tk.Scale(root, width=10, bg="#000000", fg="white", from_=1, to=10, orient="horizontal")
soil_entry.grid(row=0, column=1)

tk.Label(root, text="Water (in liters):", font=("Helvetica", 12), bg="#000000", fg="white").grid(row=1, column=0)
water_entry = tk.Scale(root, width=10, bg="#000000", fg="white", from_=0.5, to=3.0, resolution=0.1, orient="horizontal")
water_entry.grid(row=1, column=1)

tk.Label(root, text="Sunlight (in hours):", font=("Helvetica", 12), bg="#000000", fg="white").grid(row=2, column=0)
sunlight_entry = tk.Scale(root, width=10, bg="#000000", fg="white", from_=1, to=24, orient="horizontal")
sunlight_entry.grid(row=2, column=1)

# Compile the model
model.compile(optimizer='adamax', loss='mse')

# Train the model
model.fit(X, y, epochs=150, verbose=0)

os.system('clear' if sys == 'linux' else 'cls')

def predict():
    soilentered = float(float(soil_entry.get())/10)
    # Predict plant health for a given environment
    test_sample = np.array([[float(sunlight_entry.get()), float(water_entry.get()), float(soilentered)]])  # 8 hours sunlight, 2L water, good soil quality
    predicted_health = model.predict(test_sample)

    os.system('clear' if sys == 'linux' else 'cls')

    if predicted_health[0][0] < 5:
        predict_label.config(fg="red")
    elif predicted_health[0][0] < 7:
        predict_label.config(fg="yellow")
    else:
        predict_label.config(fg="green")

    if predicted_health[0][0] > 10:
        predicted_health[0][0] = 10
    
    predict_label.config(text=f"Predicted Plant Health: {predicted_health[0][0]:.2f}")

predict_label = tk.Label(root, bg="#000000", fg="white", font=("Helvetica", 12))
predict_label.grid(row=5, column=0, columnspan=2)

tk.Button(root, text="Predict", command=predict, bg="#000000", fg="white").grid(row=3, column=0, columnspan=2)
tk.Button(root, text="Quit", command=root.quit, bg="#000000", fg="white").grid(row=4, column=0, columnspan=2)

root.mainloop()

os.system('clear' if sys == 'linux' else 'cls')