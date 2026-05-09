import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge, Info
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time
import random
import psutil
import threading

# ---------------------------------------------------------------------------
# PROMETHEUS METRICS CONFIGS
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("http_requests_total", "Total requests to the model", ["method", "endpoint"])
FRAUD_DETECTED = Counter("fraud_detected_total", "Total detected fraud transactions")
NORMAL_DETECTED = Counter("normal_detected_total", "Total detected normal transactions")
ERROR_RATE = Counter("http_error_responses_total", "Total error responses (4xx/5xx)", ["status_code"])
MODEL_LATENCY = Histogram("model_prediction_latency_seconds", "Model prediction latency time (seconds)", buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0])
MEMORY_USAGE = Gauge("system_memory_usage_bytes", "System memory usage (bytes)")
CPU_USAGE = Gauge("system_cpu_usage_percent", "System CPU usage (%)")
BATCH_SIZE = Histogram("prediction_batch_size", "Number of rows per prediction request", buckets=[1, 5, 10, 50, 100, 500])
ACTIVE_SESSIONS = Gauge("active_user_sessions", "Number of active inference sessions")
ROWS_PROCESSED = Counter("rows_processed_total", "Total number of rows processed by the model")
MODEL_INFO = Info("model", "Running Model Information")

MODEL_INFO.info({"name": "FraudModel", "version": "1", "framework": "scikit-learn", "developer": "Fierda"})
_active_lock = threading.Lock()
_active_count = 0

# ---------------------------------------------------------------------------
# FLASK & MLFLOW CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_URI = os.environ.get("MODEL_URI", "models:/FraudModel/1")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
PORT = int(os.environ.get("PORT", 5001))

app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

model = None

def get_system_metrics():
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return cpu, mem.used
    except Exception:
        return 0.0, 0

def load_model():
    global model
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
        print(f"[INFO] Tracking URI: {TRACKING_URI}")

    try:
        print(f"[INFO] Loading model from: {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("[OK] Model successfully loaded!")
    except Exception as e:
        print(f"[WARN] Failed to load from MLflow registry: {e}")
        print("[INFO] Trying to load from local path...")
        local_paths = [
            "mlruns",
            os.path.join(os.path.dirname(__file__), "..", "mlruns"),
            os.path.join(os.path.dirname(__file__), "..", "Membangun_model", "mlruns"),
        ]
        for path in local_paths:
            if os.path.exists(path):
                try:
                    import glob
                    model_dirs = glob.glob(os.path.join(path, "**", "artifacts", "model", "MLmodel"), recursive=True)
                    if model_dirs:
                        model_path = os.path.dirname(model_dirs[-1])
                        model = mlflow.pyfunc.load_model(model_path)
                        print(f"[OK] Model loaded from: {model_path}")
                        return
                except Exception:
                    continue
        print("[WARN] No MLflow model found. API will return 503 until a valid model is loaded.")
        model = None


@app.route("/health", methods=["GET"])
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    return jsonify({
        "status": "healthy" if model is not None else "no_model",
        "model_uri": MODEL_URI,
    })

@app.route("/predict", methods=["POST"])
def predict():
    global _active_count
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    with _active_lock:
        _active_count += 1
        ACTIVE_SESSIONS.set(_active_count)

    if model is None:
        ERROR_RATE.labels(status_code="503").inc()
        with _active_lock:
            _active_count -= 1
            ACTIVE_SESSIONS.set(_active_count)
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if "dataframe_split" in data:
            df = pd.DataFrame(**data["dataframe_split"])
        elif "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            ERROR_RATE.labels(status_code="400").inc()
            with _active_lock:
                _active_count -= 1
                ACTIVE_SESSIONS.set(_active_count)
            return jsonify({"error": "Invalid input format."}), 400

        predictions = model.predict(df)

        BATCH_SIZE.observe(len(df))
        ROWS_PROCESSED.inc(len(df))

        fraud_count = int(sum(predictions))
        normal_count = int(len(predictions) - sum(predictions))

        if fraud_count > 0:
            FRAUD_DETECTED.inc(fraud_count)
        if normal_count > 0:
            NORMAL_DETECTED.inc(normal_count)

        result = {
            "predictions": predictions.tolist(),
            "fraud_count": fraud_count,
            "normal_count": normal_count,
        }

        latency = time.time() - start_time
        MODEL_LATENCY.observe(latency)

        cpu, mem = get_system_metrics()
        CPU_USAGE.set(cpu)
        MEMORY_USAGE.set(mem)

        # Menambahkan delay statis 0.5 detik agar request sedikit mengantre (hanya untuk memunculkan metrik Active Session)
        time.sleep(0.5)

        with _active_lock:
            _active_count -= 1
            ACTIVE_SESSIONS.set(_active_count)

        return jsonify(result)

    except Exception as e:
        ERROR_RATE.labels(status_code="500").inc()
        with _active_lock:
            _active_count -= 1
            ACTIVE_SESSIONS.set(_active_count)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Fraud Detection API",
        "developer": "Fierda",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Make predictions",
            "GET /metrics": "Prometheus Metrics (10 metrics)"
        },
        "model_uri": MODEL_URI,
    })


if __name__ == "__main__":
    load_model()
    print(f"\n[SERVER] Fraud Detection API running on http://localhost:{PORT}")
    print(f"  GET  /         → API Info")
    print(f"  GET  /health   → Health check")
    print(f"  POST /predict  → Fraud Prediction")
    print(f"  GET  /metrics  → Prometheus metrics (10 metrics)")
    
    psutil.cpu_percent()
    
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
