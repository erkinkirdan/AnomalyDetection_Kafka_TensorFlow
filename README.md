# Anomaly Detection with Kafka and TensorFlow

This repository contains a set of scripts that generate synthetic sensor data, use a pre-trained TensorFlow model to detect anomalies in the data, and evaluate the model's performance.
It demonstrates a streaming data pipeline using Apache Kafka and TensorFlow.

## Installation

### Apache Kafka Setup

1. Download the latest Kafka release and extract it: `tar -xzf kafka_2.13-3.5.0.tgz`
2. Navigate into the Kafka directory: `cd kafka_2.13-3.5.0`
3. Start the ZooKeeper server: `bin/zookeeper-server-start.sh config/zookeeper.properties`
4. Start the Kafka server: `bin/kafka-server-start.sh config/server.properties`
5. Create the Kafka topic: `kafka-topics.sh --create --topic sensor_data --bootstrap-server localhost:9092`

### Python Modules

1. Clone the repository: `git clone https://github.com/erkinkirdan/AnomalyDetection_Kafka_TensorFlow.git`
2. Navigate into the cloned repository: `cd AnomalyDetection_Kafka_TensorFlow`
3. Create a virtual environment: `python -m venv env`
4. Activate the virtual environment: `source env/bin/activate`
5. Install the necessary Python modules using pip: `pip install tensorflow confluent-kafka sklearn matplotlib`

## Running

### Train the Anomaly Detection Model

Run the `train.py` script to train the anomaly detection model and save it to disk.
This script must be run only once unless you want to retrain the model.

```bash
python train.py
```

### Consume Sensor Data and Make Predictions

Run the `consumer.py` script to start consuming the sensor data from Kafka and making predictions using the pre-trained model.
The predictions will be written in a CSV file.

```bash
python consumer.py
```

### Generate and Send Sensor Data

Run the `producer.py` script in a separate terminal to generate synthetic sensor data and send it to a Kafka topic.

```bash
python producer.py
```

### Evaluate the Model and Plot Latencies

After running the `producer.py` and `consumer.py` scripts, stop them by pressing Ctrl+C.
Then, run the `eval.py` script to evaluate the performance of the anomaly detection model and plot the latencies.

```bash
python eval.py
```
