import json
import random
import time
from confluent_kafka import Producer
from datetime import datetime

def generate_sensor_data():
    """Generate random sensor data for temperatures and humidity."""
    temp1 = random.uniform(0, 50)
    temp2 = random.uniform(0, 50)
    humidity = random.uniform(0, 100)
    timestamp = datetime.now().isoformat()

    return {
        'Temperature1': temp1,
        'Temperature2': temp2,
        'Humidity': humidity,
        'Timestamp': timestamp
    }

def main():
    # Kafka configuration
    conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(conf)

    try:
        while True:
            # Generate and send sensor data to Kafka
            message = json.dumps(generate_sensor_data())
            producer.produce('sensor_data', message)
            producer.flush()
            time.sleep(0.1)  # adjust sleep time to control data generation rate

    except KeyboardInterrupt:
        print('Stopped.')

if __name__ == '__main__':
    main()
