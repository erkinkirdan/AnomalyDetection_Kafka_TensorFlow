import csv
import json
import numpy as np
import tensorflow as tf
from confluent_kafka import Consumer
from datetime import datetime
import os

# Check if the CSV file exists, if so, delete it
if os.path.exists('labels.csv'):
    os.remove('labels.csv')

# Load the standardization parameters
with open('standardization_params.json', 'r') as f:
    params = json.load(f)
means = params['means']
stdevs = params['stdevs']

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='anomaly_detection_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Hardcoded normal ranges
normal_ranges = {
    'Temperature1': (1, 46),
    'Temperature2': (4, 49),
    'Humidity': (0, 90)
}

# Kafka configuration
conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'group1'}
consumer = Consumer(conf)
consumer.subscribe(['sensor_data'])

# CSV file setup
with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Temperature1", "Temperature2", "Humidity", "Predicted_Label", "True_Label", "Timestamp", "Kafka_Timestamp", "Prediction_Timestamp"])

try:
    while True:
        msg = consumer.poll(0)
        if msg is None:
            continue
        if msg.error():
            raise Exception(msg.error())
        else:
            # Get the Kafka timestamp
            kafka_timestamp = datetime.now().isoformat()

            # Get the message value and convert it to a dictionary
            sensor_data = json.loads(msg.value().decode('utf-8'))

            # Extract and standardize the sensor data
            data = [sensor_data['Temperature1'], sensor_data['Temperature2'], sensor_data['Humidity']]
            data = [(data[i] - means[i]) / stdevs[i] for i in range(len(data))]
            data = np.array(data, dtype=np.float32).reshape(1, -1)

            # Set the value of the input tensor
            interpreter.set_tensor(input_details[0]['index'], data)
            # Run the inference
            interpreter.invoke()
            # Retrieve the output of the model
            output = interpreter.get_tensor(output_details[0]['index'])
            # Get prediction timestamp
            prediction_timestamp = datetime.now().isoformat()

            # Determine if the data point is an anomaly
            true_label = (sensor_data['Temperature1'] < normal_ranges['Temperature1'][0] or
                          sensor_data['Temperature1'] > normal_ranges['Temperature1'][1] or
                          sensor_data['Temperature2'] < normal_ranges['Temperature2'][0] or
                          sensor_data['Temperature2'] > normal_ranges['Temperature2'][1] or
                          sensor_data['Humidity'] < normal_ranges['Humidity'][0] or
                          sensor_data['Humidity'] > normal_ranges['Humidity'][1])
            true_label = 1 if true_label else 0

            # Write sensor data, labels and timestamps to CSV
            with open('labels.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([sensor_data['Temperature1'], sensor_data['Temperature2'], sensor_data['Humidity'],
                                 int(output[0][0] > 0.5), true_label, sensor_data['Timestamp'], kafka_timestamp, prediction_timestamp])

except KeyboardInterrupt:
    print('Stopped.')
finally:
    # Close down consumer to commit final offsets
    consumer.close()
