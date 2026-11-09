
import streamlit as st
import pandas as pd
import json
import joblib
import random

from datetime import datetime
from pathlib import Path

from live_feature_extractor import extract_pcap_windows, MODEL_PATH
from streamlit_autorefresh import st_autorefresh
from mqtt_listener import start_listener, get_latest_reading


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IoT Digital Twin IDS",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path("data")

SUP = ROOT / "supervised/file_based_mitm60k"
RESULTS = SUP / "results"
ROBUST = SUP / "results_robust"


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🛡️ IoT Intrusion Detection System")

st.subheader("Digital Twin Monitoring Dashboard")

st.markdown(
    "Integrated monitoring of simulated IoT sensor behaviour "
    "and CICIoT2023-based intrusion detection experiments."
)


# --------------------------------------------------
# 1. LIVE IOT DIGITAL TWIN MONITORING
# --------------------------------------------------

st.header("1. Live IoT Digital Twin Monitoring")

mode = st.radio(
    "Monitoring Mode",
    ["Local MQTT Mode", "Cloud Demo Mode"],
    horizontal=True
)

st_autorefresh(
    interval=2000,
    key="sensor_refresh"
)

reading = None

if mode == "Cloud Demo Mode":

    temperature = round(random.uniform(24, 29), 2)
    humidity = round(random.uniform(45, 60), 2)

    reading = {
        "temperature": temperature,
        "humidity": humidity,
        "device_id": "SIM-IOT-001",
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    st.info(
        "Cloud Demo Mode: Displaying simulated IoT sensor readings."
    )

else:

    try:
        start_listener()
        reading = get_latest_reading()

        if not reading:
            st.warning(
                "Waiting for MQTT sensor data. "
                "Start sensor_publisher.py."
            )

    except Exception as error:
        st.error(f"MQTT connection issue: {error}")


if reading:

    temperature = float(reading["temperature"])
    humidity = float(reading["humidity"])

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Temperature",
        f"{temperature:.2f} °C"
    )

    c2.metric(
        "Humidity",
        f"{humidity:.2f} %"
    )

    temp_alert = abs(temperature - 26.0) > 5.0
    humidity_alert = abs(humidity - 52.0) > 15.0

    if temp_alert or humidity_alert:
        c3.error("SENSOR ANOMALY")
    else:
        c3.success("NORMAL")

    st.write(
        "Device ID:",
        reading.get("device_id", "Unknown")
    )

    st.write(
        "Timestamp:",
        reading.get("timestamp", "Unknown")
    )

    st.info(
        "Sensor anomaly status uses illustrative temperature "
        "and humidity thresholds. It is separate from ML-based "
        "network intrusion predictions."
    )

st.divider()


# --------------------------------------------------
# 2. SUPERVISED HYBRID IDS RESULTS
# --------------------------------------------------

st.header("2. Supervised Hybrid DT–LR Intrusion Detection")

try:

    with open(
        RESULTS / "hybrid_dt_lr_mitm60k_no_https_metrics.json"
    ) as f:
        original = json.load(f)

    with open(
        ROBUST / "hybrid_dt_lr_mitm60k_robust_no_https_metrics.json"
    ) as f:
        robust = json.load(f)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{original['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{original['precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{original['recall'] * 100:.2f}%"
    )

    c4.metric(
        "F1-score",
        f"{original['f1_score'] * 100:.2f}%"
    )

    st.metric(
        "ROC-AUC",
        f"{original['roc_auc'] * 100:.2f}%"
    )

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ],

        "Original Hybrid": [
            original["accuracy"] * 100,
            original["precision"] * 100,
            original["recall"] * 100,
            original["f1_score"] * 100,
            original["roc_auc"] * 100
        ],

        "RobustScaler Hybrid": [
            robust["accuracy"] * 100,
            robust["precision"] * 100,
            robust["recall"] * 100,
            robust["f1_score"] * 100,
            robust["roc_auc"] * 100
        ]

    })

    st.subheader("Original Hybrid vs RobustScaler")

    st.bar_chart(
        comparison.set_index("Metric")
    )

    st.dataframe(
        comparison.round(2),
        use_container_width=True
    )

    st.write(
        f"Original training time: "
        f"{original['training_time_seconds']:.2f} seconds"
    )

    st.write(
        f"RobustScaler training time: "
        f"{robust['training_time_seconds']:.2f} seconds"
    )

    st.success(
        "Original Hybrid DT–LR retained as the main candidate."
    )

except Exception as error:

    st.error(
        f"Unable to load supervised metrics: {error}"
    )

st.divider()


# --------------------------------------------------
# 3. MITM ATTACK ANALYSIS
# --------------------------------------------------

st.header("3. MITM Attack Detection Analysis")

try:

    prediction_path = (
        RESULTS / "hybrid_dt_lr_mitm60k_no_https_predictions.csv"
    )

    predictions = pd.read_csv(prediction_path)

    mitm = predictions[
        predictions["attack_category"] == "MITM-ArpSpoofing"
    ]

    if len(mitm) > 0:

        detected = int(
            (mitm["predicted"] == 1).sum()
        )

        missed = int(
            (mitm["predicted"] == 0).sum()
        )

        rate = detected / len(mitm) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "MITM Test Records",
            f"{len(mitm):,}"
        )

        c2.metric(
            "Detected",
            f"{detected:,}"
        )

        c3.metric(
            "Detection Rate",
            f"{rate:.2f}%"
        )

        st.write(
            "Missed MITM records:",
            missed
        )

        st.bar_chart(
            pd.DataFrame(
                {
                    "Records": [detected, missed]
                },
                index=["Detected", "Missed"]
            )
        )

    else:

        st.warning(
            "No MITM-ArpSpoofing records found."
        )

except Exception as error:

    st.error(
        f"Unable to load MITM results: {error}"
    )

st.divider()


# --------------------------------------------------
# 4. ATTACK CATEGORY COMPARISON
# --------------------------------------------------

st.header("4. Attack Category Comparison")

try:

    category_path = SUP / "category_comparison.csv"

    categories = pd.read_csv(category_path)

    st.write(
        "Comparison of MITM training experiments using "
        "3K, 30K and 60K training records."
    )

    st.bar_chart(
        categories.set_index("attack_category")[
            ["3K", "30K", "60K"]
        ]
    )

    st.dataframe(
        categories.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This file contains 18 independently held-out attack "
        "categories, not all 33 categories."
    )

except Exception as error:

    st.error(
        f"Unable to load category comparison: {error}"
    )

st.divider()


# --------------------------------------------------
# 5. UNSUPERVISED DIGITAL TWIN RESULTS
# --------------------------------------------------

st.header("5. Unsupervised Digital Twin Anomaly Detection")

DATA_DIR = ROOT / "processed/splits"

try:

    attacks = pd.read_csv(
        DATA_DIR / "attack_detection_results.csv"
    )

    baseline = pd.read_csv(
        DATA_DIR / "baseline_stats_transformed.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "benign_validation.csv"
    )

    robust_rate = attacks[
        "robust_detection_rate"
    ].mean()

    mahal_rate = attacks[
        "mahal_detection_rate"
    ].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Attack Categories",
        len(attacks)
    )

    c2.metric(
        "Benign Validation Records",
        f"{len(validation):,}"
    )

    c3.metric(
        "Selected Baseline Features",
        len(baseline)
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Robust Z-score Macro Detection",
        f"{robust_rate:.2f}%"
    )

    c2.metric(
        "Mahalanobis Macro Detection",
        f"{mahal_rate:.2f}%"
    )

    st.subheader("Benign Digital Twin Baseline")

    st.dataframe(
        baseline,
        use_container_width=True
    )

    st.subheader("Detection by Attack Category")

    chart = attacks.set_index("attack_category")[
        [
            "robust_detection_rate",
            "mahal_detection_rate"
        ]
    ]

    st.bar_chart(chart)

    st.dataframe(
        attacks.sort_values(
            "mahal_detection_rate"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Benign validation uses held-out CICIoT2023 records."
    )

except Exception as error:

    st.error(
        f"Unsupervised results unavailable: {error}"
    )

st.divider()


# --------------------------------------------------
# 6. PCAP-BASED NETWORK IDS INFERENCE
# --------------------------------------------------

st.header("6. PCAP-Based Network IDS Inference")


@st.cache_resource
def load_live_ids_model():

    return joblib.load(MODEL_PATH)


capture_path = Path("captures/iot_mqtt.pcapng")

st.write(
    "Capture file:",
    str(capture_path)
)

if st.button(
    "Run PCAP IDS Analysis",
    key="run_pcap_ids"
):

    try:

        if not capture_path.exists():

            st.error(
                "PCAP capture file not found."
            )

        else:

            with st.spinner(
                "Extracting features and running IDS predictions..."
            ):

                features = extract_pcap_windows(
                    capture_path
                )

                if features.empty:

                    st.warning(
                        "No valid packet windows were extracted."
                    )

                else:

                    model = load_live_ids_model()

                    predictions = model.predict(
                        features
                    )

                    probabilities = (
                        model.predict_proba(features)
                        if hasattr(model, "predict_proba")
                        else None
                    )

                    benign_count = int(
                        (predictions == 0).sum()
                    )

                    attack_count = int(
                        (predictions == 1).sum()
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Total Windows",
                        len(predictions)
                    )

                    c2.metric(
                        "Benign Predictions",
                        benign_count
                    )

                    c3.metric(
                        "Attack Predictions",
                        attack_count
                    )

                    if attack_count > 0:

                        st.error(
                            "Network attack predictions detected."
                        )

                    else:

                        st.success(
                            "All analysed windows predicted benign."
                        )

                    result_data = {

                        "Window": range(
                            1,
                            len(predictions) + 1
                        ),

                        "Prediction": [
                            "Attack" if int(value) == 1 else "Benign"
                            for value in predictions
                        ]

                    }

                    if probabilities is not None:

                        classes = list(
                            model.classes_
                        )

                        if 1 in classes:

                            attack_index = classes.index(1)

                            result_data[
                                "Attack Probability (%)"
                            ] = [

                                round(
                                    float(value) * 100,
                                    2
                                )

                                for value in probabilities[
                                    :, attack_index
                                ]

                            ]

                    results_df = pd.DataFrame(
                        result_data
                    )

                    st.subheader(
                        "Window-by-Window IDS Results"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Class 0 = Benign; Class 1 = Attack. "
                        "These are predictions on the selected PCAP "
                        "capture, not a measurement of live network "
                        "detection accuracy."
                    )

    except Exception as error:

        st.error(
            f"PCAP IDS analysis failed: {error}"
        )

st.divider()


# --------------------------------------------------
# 7. SYSTEM STATUS
# --------------------------------------------------

st.header("7. System Status")

st.write(
    "✔ CICIoT2023 supervised models trained"
)

st.write(
    "✔ Hybrid DT–LR evaluation saved"
)

st.write(
    "✔ MQTT sensor publisher and listener implemented"
)

st.write(
    "✔ Digital Twin sensor monitoring implemented"
)

st.write(
    "✔ Offline network packet capture and inference available"
)

st.warning(
    "Cloud Demo Mode uses simulated sensor readings. "
    "Local MQTT Mode receives readings from the local MQTT broker. "
    "Live network traffic has not yet been validated against the "
    "trained 34-feature IDS model. Offline model metrics and "
    "live sensor alerts are shown separately."
)

st.caption(
    "IoT Intrusion Detection with Digital Twin Modeling"
)

