# IoT Intrusion Detection System with Digital Twin Modeling

A software-based IoT Intrusion Detection System (IDS) integrating supervised machine learning, Digital Twin anomaly monitoring, MQTT sensor simulation, and PCAP-based network traffic analysis.

## Project Overview

This project explores an integrated approach to IoT security by combining machine learning-based intrusion detection with Digital Twin modeling.

It uses the CICIoT2023 dataset for intrusion detection experiments and simulated MQTT sensor readings for Digital Twin monitoring.

## Key Features

- **Hybrid Machine Learning IDS:** Combines Decision Tree and Logistic Regression using a Voting Classifier.
- **Digital Twin Monitoring:** Monitors simulated IoT sensor behaviour using benign baseline statistics.
- **Anomaly Detection:** Uses Robust Z-score and Mahalanobis distance.
- **PCAP Traffic Analysis:** Extracts network traffic features from packet capture files.
- **MQTT Integration:** Receives simulated IoT sensor readings through MQTT.
- **Streamlit Dashboard:** Provides a unified interface for monitoring and experiment results.
- **Model Evaluation:** Supports metrics and attack-category comparisons.

## Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- Streamlit
- MQTT / Paho-MQTT
- dpkt
- Joblib

## Dataset

This project uses the **CICIoT2023 dataset**.

The original dataset is not included in this repository because of its size. Download it separately and place the required CSV files inside the project's `data/raw/` directory.

Dataset source: https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Installation

Clone the repository:

```bash
git clone https://github.com/AbithaFelicia/iot.git
cd iot
