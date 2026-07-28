
# Unified ML Model Manager & Inference Server

A dual-purpose Python application designed to manage, switch, and serve machine learning models via a unified interface. It provides an administrative **NiceGUI Dashboard** for model management and a high-performance **FastAPI REST API** for production-style inference.

## 🚀 Key Features

* **Framework Agnostic:** A single implementation for Scikit-learn, PyTorch, and TensorFlow using the Adapter Design Pattern.
* **Administrative Dashboard:** A reactive web interface to:
    * Upload new models.
    * Persist models to permanent storage.
    * Select/Swap active models instantly without server restarts.
    * Monitor system status via dynamic color-coded visual feedback.
* **High-Performance API:** A standardized REST API for high-speed predictions and probability scores.
* **Secure Inference:** Built-in token-based authentication for all API requests.

---

## 🏗 Architecture Overview

The project follows a modular, decoupled architecture:

1.  **`core.py` (The Engine):** Defines an `Abstract Base Class (ABC)` that forces all model wrappers to implement `.predict()` and `.predict_proba()`. This ensures the API doesn't care if it is talking to a Neural Network or a Random Forest.
2.  **`app.py` (The Controller):** 
    * **FastAPI Layer:** Handles high-speed JSON requests, performs feature vectorization (alphabetical sorting), and enforces security.
    * **NiceGUI Layer:** Manteins the "Global State." When you change a model in the UI, it updates the shared memory that the API uses.

---

## 🛠 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install nicegui fastapi uvicorn numpy scikit-learn torch tensorflow joblib
```

### 3. Environment Configuration
For security, the API requires a token. You can set this in your environment:

**Linux/macOS:**
```bash
export API_SECRET_TOKEN="your_super_secret_token"
```

**Windows (PowerShell):**
```powershell
$env:API_SECRET_TOKEN="your_super_secret_token"
```
*(Note: If not set, the system defaults to `pass123` for local development.)*

### 4. Run the Application
```bash
python app.py
```
The server will start at `http://localhost:8000`.

---

## 📖 Usage Guide

### A. Management (Web UI)
1.  **Navigate to:** `http://localhost:8000` in your browser.
2.  **Upload:** Drag and drop a model file (`.pkl`, `.pt`, or `.h5`) into the Upload Card.
3.  **Save:** Once uploaded, select it from the dropdown and click **"Save Permanently"**. This moves the file to the `/models` directory for persistence across restarts.
4.  **Activate:** Select a model from the "Available Models" list. 
    *   **Green Background/Indicator:** Model is active and ready for API calls.
    *   **Red Background/Indicator:** System is idle; no model is currently serving requests.
5.  **Un-ready:** Click **"Un-ready Model"** to clear the selection and return the system to a safe, idle state.

### B. Inference (API via Curl)
The API expects a JSON payload containing your `token` and a dictionary of features (`datapoint`). The engine automatically sorts these features alphabetically before passing them to the model.

**Predict Class:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "token": "your_super_secret_token",
           "datapoint": {
             "EGA": [29.9],
             "BWGT": [1430.0],
             "SEX": [1.0]
           }
         }'
```

**Predict Probabilities:**
```bash
curl -X POST "http://localhost:8000/predict_proba" \
     -H "Content-Type: application/json" \
     -d '{
           "token": "your_super_secret_token",
           "datapoint": { "EGA": [29.9], "BWGT": [1430.0], "SEX": [1.0] }
         }'
```

---

## 📁 Project Structure
* `app.py`: The entry point containing the FastAPI routes and NiceGUI dashboard logic.
* `core.py`: The core engine containing model wrappers, the loader factory, and global state management.
* `/models/`: Permanent storage for saved models.
* `/temp_models/`: Temporary staging area for uploaded files.

## ⚠️ Security & Production Notes
* **Token Security:** In production, always use a long, complex string via environment variables. Never hardcode secrets in `app.py`.
* **Scaling:** This application is designed as a single-process "Control Center." For massive scale (thousands of requests per second), consider separating the NiceGUI management tool from a dedicated FastAPI cluster.