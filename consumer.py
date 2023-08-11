import csv
import json
import numpy as np
import tflite_runtime.interpreter as tflite
from confluent_kafka import Consumer
from datetime import datetime
import os

def check_remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def load_params(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def kafka_config(servers, group_id):
    return Consumer({'bootstrap.servers': servers, 'group.id': group_id})

def write_csv_header(file_name, header):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def standardize_data(sensor_data, means, stdevs):
    return np.array([(sensor_data[i] - means[i]) / stdevs[i] for i in range(len(sensor_data))], dtype=np.float32).reshape(1, -1)

def is_anomaly(sensor_data, normal_ranges):
    return 1 if any(sensor_data[key] < normal_ranges[key][0] or sensor_data[key] > normal_ranges[key][1] for key in normal_ranges) else 0

def main():
    check_remove_file('labels.csv')

    params = load_params('standardization_params.json')
    means, stdevs = params['means'], params['stdevs']
    interpreter = load_model('anomaly_detection_model.tflite')
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

    normal_ranges = {
        'Temperature1': (1, 46),
        'Temperature2': (4, 49),
        'Humidity': (0, 90)
    }

    consumer = kafka_config('localhost:9092', 'group1')
    consumer.subscribe(['sensor_data'])
    write_csv_header('labels.csv', ["Temperature1", "Temperature2", "Humidity", "Predicted_Label", "True_Label", "Timestamp", "Kafka_Timestamp", "Prediction_Timestamp"])

    try:
        while True:
            msg = consumer.poll(0)
            if msg is None:
                continue
            if msg.error():
                raise Exception(msg.error())

            kafka_timestamp = datetime.now().isoformat()
            sensor_data = json.loads(msg.value().decode('utf-8'))
            data = standardize_data([sensor_data['Temperature1'], sensor_data['Temperature2'], sensor_data['Humidity']], means, stdevs)
            interpreter.set_tensor(input_details[0]['index'], data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            prediction_timestamp = datetime.now().isoformat()
            true_label = is_anomaly(sensor_data, normal_ranges)

            with open('labels.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([sensor_data['Temperature1'], sensor_data['Temperature2'], sensor_data['Humidity'],
                                 int(output[0][0] > 0.5), true_label, sensor_data['Timestamp'], kafka_timestamp, prediction_timestamp])

    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
