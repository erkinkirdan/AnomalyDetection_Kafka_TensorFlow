# Anomaly Detection with Kafka and TensorFlow

This repository contains a set of scripts that generate synthetic sensor data, use a pre-trained TensorFlow model to detect anomalies in the data, and evaluate the model's performance.

It demonstrates a streaming data pipeline using Apache Kafka and TensorFlow.

## Installation

1. **Python Modules**

    Install the necessary Python modules using pip:

    ```bash
    pip install tensorflow confluent-kafka sklearn matplotlib
    ```

2. **Apache Kafka Setup**

    Start the ZooKeeper and Kafka servers:

    ```bash
    # Start ZooKeeper
    zookeeper-server-start.sh /usr/local/etc/kafka/zookeeper.properties

    # Start Kafka
    kafka-server-start.sh /usr/local/etc/kafka/server.properties
    ```

    Create the Kafka topic:

    ```bash
    kafka-topics.sh --create --topic sensor_data --bootstrap-server localhost:9092
    ```

## Running

1. **Train the Anomaly Detection Model**

    Run the `train.py` script to train the anomaly detection model and save it to disk. This script needs to be run only once, unless you want to retrain the model.

    ```bash
    python train.py
    ```

2. **Consume Sensor Data and Make Predictions**

    Run the `consumer.py` script to start consuming the sensor data from Kafka and making predictions using the pre-trained model. The predictions will be written to a CSV file.

    ```bash
    python consumer.py
    ```

3. **Generate and Send Sensor Data**

    In a separate terminal, run the `producer.py` script to start generating synthetic sensor data and sending it to a Kafka topic.

    ```bash
    python producer.py
    ```

4. **Evaluate the Model and Plot Latencies**

    After running the `producer.py` and `consumer.py` scripts for a while, stop them by pressing Ctrl+C. Then, run the `eval.py` script to evaluate the performance of the anomaly detection model and plot the latencies.

    ```bash
    python eval.py
    ```
