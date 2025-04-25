import numpy as np, random, os, time
import tensorflow as tf, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

os.system('cls' if os.name == 'nt' else 'clear')

# Generate authorized MAC as a list of 6 integers (0-255) with fixed prefix 00:11:22
def generate_authorized_mac():
    prefix = [0, 17, 34]  # 00, 11, 22 in decimal
    suffix = [random.randint(0, 255) for _ in range(3)]
    return prefix + suffix

# Generate unauthorized MAC ensuring prefix is not 00:11:22
def generate_unauthorized_mac():
    while True:
        mac = [random.randint(0, 255) for _ in range(6)]
        if mac[:3] != [0, 17, 34]:
            return mac

# Create dataset
n_samples = 10000
authorized = [generate_authorized_mac() for _ in range(n_samples)]
unauthorized = [generate_unauthorized_mac() for _ in range(n_samples)]

# Labels: 0 for authorized, 1 for unauthorized
X = np.array(authorized + unauthorized, dtype=np.float32)
y = np.array([0]*n_samples + [1]*n_samples, dtype=np.float32)

# Normalize each octet to [0,1]
X = X / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

os.system('clear')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", accuracy)

# Save the model
if accuracy > 0.90:
    model.save("model.h5")

time.sleep(1)
os.system('clear')
exit()