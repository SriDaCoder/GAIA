import tensorflow as tf
import numpy as np, streamlit as st, io, os, sys
"""
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential"""
from sklearn.preprocessing import MinMaxScaler
# DO NOT REMOVE OR MODIFY THE FOLLOWING CODE
# This is required to ensure that your ML model, as well as the streamlit app, run as expected
a = ""
os.system('cls' if sys.platform.startswith('win') else 'clear')
# Generate synthetic training data
np.random.seed(42)
num_samples = 1000
# Features: pH (0-14), turbidity (0-100 NTU), dissolved oxygen (0-14 mg/L), contaminants (0-100 scale)
x_train = np.random.rand(num_samples, 4) * [14, 100, 14, 100]
y_train = np.clip(14 - (x_train[:, 1] / 10) - (x_train[:, 3] / 10) + (x_train[:, 2] / 2) - abs(x_train[:, 0] - 7), 1, 10)
# Normalize data
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
# Build the AI model
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
def estimate_water_usability(pH, turbidity, dissolved_oxygen, contaminants):
    st.write("Predicting (Please wait!!😊)...")
    input_data = np.array([[pH, turbidity, dissolved_oxygen, contaminants]])
    input_scaled = scaler.transform(input_data)
    usability_score = model.predict(input_scaled)[0][0]
    return round(np.clip(usability_score, 1, 14), 2)
# Streamlit UI
st.title("Water Usability Estimator")
pH = st.slider("pH Level", 0.0, 14.0, 7.0, on_change=None)
turbidity = st.slider("Turbidity (NTU)", 0, 100, 10, on_change=None)
dissolved_oxygen = st.slider("Dissolved Oxygen (mg/L)", 0.0, 14.0, 8.0, on_change=None)
contaminants = st.slider("Contaminants (scale 0-100)", 0, 100, 20, on_change=None)
if st.button("Estimate Usability"):
    result = estimate_water_usability(pH, turbidity, dissolved_oxygen, contaminants)
    st.write(f"Estimated Water Usability Score: {result}")
if st.checkbox("Model Summary") == True:
    # Capture the model summary as a string
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    buffer = buffer.getvalue().replace("━┳━", "").replace("━┻━", "").replace("───", "").replace("├", "|").replace("└", "|").replace("┌", "|").replace("│", "|").replace("─", "-").replace("┼", "|").replace("━", "").replace("┓ ", "").replace("┃", "").replace("┩", "").replace("╇", "").replace("┏", "").replace("┓", "").replace("|", "").replace("-", "").replace("┘", "").replace("┤", "").replace("┡", "").replace("#", "").replace("┴", "")
    buffer = buffer.split("\n")
    # Now you can use st.write to display the summary
    for i in range(len(buffer)):
        if "layer(type)outputshapeparam" in buffer[i].strip().lower():
            a += f"{buffer[i]}: \n"
        else:
            a += f"{buffer[i]}\n"
    st.write(a)