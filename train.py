import random
import tensorflow as tf
import json

# Define the normal and anomaly ranges
NORMAL_RANGES = {
    'Temperature1': (1, 46),
    'Temperature2': (4, 49),
    'Humidity': (0, 90)
}

def generate_random_samples(num_samples):
    temp1 = [random.uniform(0, 50) for _ in range(num_samples)]
    temp2 = [random.uniform(0, 50) for _ in range(num_samples)]
    humidity = [random.uniform(0, 100) for _ in range(num_samples)]
    return list(zip(temp1, temp2, humidity))

def create_labels(data):
    return [1 if (t1 < NORMAL_RANGES['Temperature1'][0] or t1 > NORMAL_RANGES['Temperature1'][1] or 
                  t2 < NORMAL_RANGES['Temperature2'][0] or t2 > NORMAL_RANGES['Temperature2'][1] or
                  h < NORMAL_RANGES['Humidity'][0] or h > NORMAL_RANGES['Humidity'][1]) else 0 
            for t1, t2, h in data]

def standardize(dataset):
    dimensions = len(dataset[0])
    means = [sum([row[i] for row in dataset]) / len(dataset) for i in range(dimensions)]
    stdevs = [((sum([(row[i] - means[i]) ** 2 for row in dataset]) / (len(dataset) - 1)) ** 0.5) for i in range(dimensions)]
    return [[(row[i] - means[i]) / stdevs[i] for i in range(dimensions)] for row in dataset], means, stdevs

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    num_samples = 10000
    data = generate_random_samples(num_samples)
    labels = create_labels(data)

    # Split the data
    split_point = int(num_samples * 0.8)
    X_train, X_test = data[:split_point], data[split_point:]
    y_train, y_test = labels[:split_point], labels[split_point:]

    # Standardize the data
    X_train, means, stdevs = standardize(X_train)
    X_test = [[(row[i] - means[i]) / stdevs[i] for i in range(len(row))] for row in X_test]

    # Save the means and stdevs
    with open('standardization_params.json', 'w') as f:
        json.dump({'means': means, 'stdevs': stdevs}, f)

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Convert and save the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('anomaly_detection_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    main()
