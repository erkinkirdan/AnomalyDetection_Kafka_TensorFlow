import random
import tensorflow as tf
import json

# Define the normal and anomaly ranges
normal_ranges = {
    'Temperature1': (1, 46),
    'Temperature2': (4, 49),
    'Humidity': (0, 90)
}

# Generate random samples
num_samples = 10000
temp1 = [random.uniform(0, 50) for _ in range(num_samples)]
temp2 = [random.uniform(0, 50) for _ in range(num_samples)]
humidity = [random.uniform(0, 100) for _ in range(num_samples)]

# Combine into a 2D list
data = list(zip(temp1, temp2, humidity))

# Create labels: 1 for anomaly, 0 for normal
labels = [(t1 < normal_ranges['Temperature1'][0] or t1 > normal_ranges['Temperature1'][1] or 
            t2 < normal_ranges['Temperature2'][0] or t2 > normal_ranges['Temperature2'][1] or
            h < normal_ranges['Humidity'][0] or h > normal_ranges['Humidity'][1]) for t1, t2, h in data]

# Convert labels to binary format
labels = [1 if label else 0 for label in labels]

# Split the data into training and testing sets
split_point = int(num_samples * 0.8)
X_train = data[:split_point]
X_test = data[split_point:]
y_train = labels[:split_point]
y_test = labels[split_point:]

# Manually standardize the data
def standardize(dataset):
    dimensions = len(dataset[0])
    means = [sum([row[i] for row in dataset]) / len(dataset) for i in range(dimensions)]
    stdevs = [((sum([(row[i] - means[i])**2 for row in dataset]) / (len(dataset)-1))**0.5) for i in range(dimensions)]
    return [[(row[i] - means[i]) / stdevs[i] for i in range(dimensions)] for row in dataset], means, stdevs

X_train, means, stdevs = standardize(X_train)
X_test = [[(row[i] - means[i]) / stdevs[i] for i in range(len(row))] for row in X_test]

# Save the means and stdevs as a JSON
with open('standardization_params.json', 'w') as f:
    json.dump({'means': means, 'stdevs': stdevs}, f)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open('anomaly_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
