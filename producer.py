import json
import random
import time
from confluent_kafka import Producer
from datetime import datetime

# Kafka configuration
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

try:
    while True:
        # Generate random sensor data
        temp1 = random.uniform(0, 50)
        temp2 = random.uniform(0, 50)
        humidity = random.uniform(0, 100)
        timestamp = datetime.now().isoformat()

        # Produce and send message to Kafka
        message = json.dumps({
            'Temperature1': temp1,
            'Temperature2': temp2,
            'Humidity': humidity,
            'Timestamp': timestamp
        })
        producer.produce('sensor_data', message)
        producer.flush()

        time.sleep(0.1)  # adjust the sleep time to control the rate of data generation

except KeyboardInterrupt:
    print('Stopped.')
