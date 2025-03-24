import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Sample training data (Replace with actual financial data)
X_train = np.random.rand(1000, 5)  # 1000 samples, 5 features (Example: Income, Savings, Expenses, etc.)
y_train = np.random.rand(1000, 1)  # 1000 labels (Example: Financial risk score)

# Model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer (Regression)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Save model in Keras format
model.save("financial_advice_model.keras")

print("Model trained and saved as 'financial_advice_model.keras'")
