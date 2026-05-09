import requests
import time
import random
import json
import threading

URL = "http://localhost:5001/predict"
COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

def generate_random_data():
    # Simulate a single row with correct feature names (V1-V28, Amount_scaled, Time_scaled)
    return {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": [[random.uniform(-5, 5) for _ in range(30)]]
        }
    }

def send_request():
    try:
        # 5% chance of sending bad data to trigger 400 error
        if random.random() < 0.05:
            requests.post(URL, json={"bad_data": "to trigger error"}, timeout=2)
            print("[x] Sent bad request (Error simulated)")
        else:
            payload = generate_random_data()
            response = requests.post(URL, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"[v] Success: {response.json()}")
            else:
                print(f"[!] Warning HTTP {response.status_code}")
    except Exception as e:
        print(f"[-] Request failed: {e}")

def run_load_test():
    print("Start Load Testing to generate Prometheus & Grafana metrics...")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Simulate heavy traffic (50-100 concurrent requests)
            threads = []
            for _ in range(random.randint(50, 100)):
                t = threading.Thread(target=send_request)
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join()
                
            # No sleep! Blast the server continuously
    except KeyboardInterrupt:
        print("\nLoad testing stopped.")

if __name__ == "__main__":
    run_load_test()
