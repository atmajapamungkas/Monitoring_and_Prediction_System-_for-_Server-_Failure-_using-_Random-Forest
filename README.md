# 🖥️ Server Health Failure Prediction

A real-time server health monitoring dashboard with machine learning-based time-to-failure prediction, built with Python, Tkinter, and scikit-learn.

## 📌 Overview

This application monitors live system metrics (CPU, memory, disk, and network latency) and uses a trained Random Forest regression model to predict the estimated time remaining before a potential server failure. Results are displayed in a desktop GUI with live-updating charts and a countdown timer.

## ✨ Features

- Real-time monitoring of CPU, memory, disk usage, and network latency
- Machine learning-based time-to-failure prediction (Random Forest Regressor)
- Live status indicator (Healthy / Stable / Warning / Critical / Failure Imminent)
- Countdown timer to predicted failure time
- 6-panel live dashboard (metrics history + health status bar)

## ⚠️ Required: Training Dataset

This project requires a training dataset named `time_to_failure_dataset.csv` in the project root, which is **not included** in this repository.

The dataset must contain:
- Feature columns matching system metrics (e.g. CPU, memory, disk, latency, and any additional features used during training)
- A target column named `time_to_failure` (in seconds) representing how long until failure occurred

**You need to supply your own historical server metrics dataset** (real or collected over time) with this structure before running the application. Without it, the app will raise an error on startup.

## 🛠️ Requirements

See `requirements.txt`. Note that `tkinter` ships with most standard Python installations and is not installed via pip — if missing, install it via your OS package manager (e.g. `sudo apt install python3-tk` on Debian/Ubuntu).

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Place your `time_to_failure_dataset.csv` file in the project root.
2. Install dependencies (see above).
3. Run the application:
```bash
python server_health_monitor.py
```

## ⚙️ Configuration Notes

- The network latency check pings `192.168.1.1` by default — update this IP in `get_latency()` to match your target host/router.
- `ping3` may require elevated/root privileges on some operating systems to send ICMP packets.
- Disk usage checks common Windows/Linux mount points (`C:\`, `D:\`, `/`) — adjust `get_disk_usage()` if your environment differs.

## ⚠️ Limitations

- Prediction quality entirely depends on the quality and relevance of the training dataset you supply.
- The model is retrained fresh each run from the CSV — no persistence/versioning of trained models is implemented.
- This is a monitoring/demo tool, not a production-grade alerting system — use dedicated infrastructure monitoring tools (Prometheus, Grafana, Datadog, etc.) for real production environments.

## 🛠️ Tools

Python · Tkinter · scikit-learn · pandas · NumPy · Matplotlib · psutil · ping3

## 📄 License

MIT License
