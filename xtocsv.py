import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Load training data
data = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\3rd\\pp\\combined_features\\combined_features_csv.csv")  # Update with correct path

# ✅ Define function to create labels based on file names
def label_from_filename(filename):
    if "GGBSGB" in filename:
        return 0  # Good
    elif "FGBSBOTHFB" in filename:
        return 1  # Fault - Inner and Outer Race
    elif "FGBSIRFB" in filename:
        return 2  # Fault - Inner Race
    elif "FGBSORFB" in filename:
        return 3  # Fault - Outer Race
    else:
        return -1  # Unknown

# ✅ Apply labeling function
data['Label'] = data['FileName'].apply(label_from_filename)

# ✅ Remove unknown labels
data = data[data['Label'] != -1]

# ✅ Separate features and labels
X = data.drop(columns=["FileName", "Label"], errors='ignore')
y = data["Label"]

# ✅ Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split data (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Define a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes: Good, Inner, Outer, Both faults
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

# ✅ Save as TensorFlow model (correct format)
model.save("bearing_fault_model.keras")  # 🔹 Fixed: Use .keras format

# ✅ Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # 🔹 Fixed: Use model directly
tflite_model = converter.convert()

# ✅ Save TFLite model
with open("bearing_fault_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and converted to TFLite successfully.")
