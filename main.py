import os
import shutil
from nicegui import ui, app, events
import numpy as np
from fastapi import HTTPException
import logging
# Import our custom engine
from core import state, ModelLoader, PredictionRequest

# initiate logging module
logger = logging.getLogger(__name__)

# Load the secret from environment variables. 
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN")
APP_PORT = os.getenv("APP_PORT")

# --- 1. FASTAPI ENDPOINTS (The API Layer) ---
@app.post("/predict")
async def api_predict(request: PredictionRequest):
    if request.token != API_SECRET_TOKEN:
        logger.error("Invalid or missing API token.")
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    if not state.active_model:
        logger.error("No model active.")
        raise HTTPException(status_code=503, detail="No model active.")
    try:
        logger.info("Started /predict function")
        sorted_keys = sorted(request.datapoint.keys())
        feature_vector = []
        cols = [request.datapoint[k] for k in sorted_keys]    # each is a length-3 list
        X = np.array(cols, dtype=float).T
            
        preds = state.active_model.predict(X)
        return {
            "predictions": preds.tolist(), 
            "framework": state.active_model.framework,
            "features_used": sorted_keys # Added for your debugging/verification
        }
        logger.info("Finished /predict function")
    except Exception as e:
        logger.error("Error performing /predict function")
        logger.error(f"{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_proba")
async def api_predict_proba(request: PredictionRequest):
    if request.token != API_SECRET_TOKEN:
        logger.error("Invalid or missing API token.")
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    if not state.active_model:
        logger.error("No model active.")
        raise HTTPException(status_code=503, detail="No model active.")
    try:
        logger.info("Started /predict_proba function")
        sorted_keys = sorted(request.datapoint.keys())
        feature_vector = []
        cols = [request.datapoint[k] for k in sorted_keys]    # each is a length-3 list
        X = np.array(cols, dtype=float).T

        probs = state.active_model.predict_proba(X)
        return {
            "probabilities": probs.tolist(), 
            "framework": state.active_model.framework,
            "features_used": sorted_keys
        }
        logger.info("Finished /predict_proba function")
    except Exception as e:
        logger.error("Error performing /predict_proba function")
        logger.error(f"{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. NICEGUI DASHBOARD (The Management Layer) ---
@ui.page('/')
def dashboard():
    # starting logging module
    logging.basicConfig(filename='logs/ml-model-inference.log', level=logging.INFO)
    logger.info('Started ML Model Inference System')
    # ensure the model paths exist
    if not os.path.exists('temp_models'): os.makedirs('temp_models')
    if not os.path.exists('models'): os.makedirs('models')

    def load_existing_models():
        for filename in os.listdir('models'):
            path = os.path.join('models', filename)
            if os.path.isfile(path):
                try:
                    wrapper = ModelLoader.load(path)
                    state.registry[filename] = wrapper
                    print(f"System Startup: Loaded saved model {filename}")
                except Exception as e:
                    print(f"System Startup: Failed to load {filename}: {e}")
        model_select.set_options(list(state.registry.keys()))
        model_select.update()
        registry_list.clear()
        with registry_list:
            for name, w in state.registry.items():
                ui.item(f"{name} [{w.framework}]")

    async def handle_upload(e: events.UploadEventArguments):
        try:
            temp_path = f"temp_models/tmp_{e.file.name}"
            await e.file.save(temp_path)
            
            wrapper = ModelLoader.load(temp_path)
            state.registry[e.file.name] = wrapper
            # Update UI
            model_select.options = list(state.registry.keys())
            model_select.update()
            registry_list.clear()
            with registry_list:
                for name, w in state.registry.items():
                    ui.item(f"{name} [{w.framework}]")
            
            ui.notify(f"Loaded {e.file.name}", type='positive')
            logger.info(f"Loaded {e.file.name} model.")
        except Exception as ex:
            logger.error(f"{str(ex)}")
            ui.notify(f"Error: {ex}", type='negative')

    def save_permanently():
        target_name = model_select.value
        if not target_name or target_name not in state.registry:
            ui.notify("No model selected to save!", type='warning')
            logger.error("No model selected to save!")
            return
        try:
            old_path = f"temp_models/tmp_{target_name}"
            new_path = f"models/{target_name}"
            
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                ui.notify(f"Saved: {target_name} to /models", type='positive')
                logger.info(f"Saved: {target_name} to /models")
            else:
                ui.notify("Error: Temporary file not found. Try re-uploading.", type='negative')
                logger.error("Error: Temporary file not found. Try re-uploading.")
        except Exception as ex:
            logger.error(f"Save failed: {str(ex)}")
            ui.notify(f"Save failed: {ex}", type='negative')

    def unready_model():
        select_model(None) # This triggers your existing 'else' logic for reset

    def select_model(name):
        if name and name in state.registry:
            state.active_model = state.registry[name]
            info_label.text = f"Status: ACTIVE ({state.active_model.framework})"
            ui.query('body').style('background-color: #22C55E') # Tailwind green-500
            main_head_label.text = "Model Active"
            ui.notify(f"Model switched to {name}")
            logger.info(f"Model switched to {name}")
        else:
            state.active_model = None
            model_select.value = None 
            info_label.text = "Status: Idle"
            ui.query('body').style('background-color: #EF4444')
            main_head_label.text = "Model not Active"
            ui.notify("Model un-readied", type='warning')
            logger.error("Model un-readied")

    # 1. MAIN VERTICAL CONTAINER
    with ui.column().classes('w-full items-center py-10'):
        # Centered Title
        main_head_label = ui.label('Model not Active').classes('text-h2 font-bold text-white mb-8')
        # 2.1 UPLOAD MODEL CARD
        with ui.row().classes('w-full max-w-6xl justify-center no-wrap'):
            with ui.card().classes('w-1/2 shadow-lg bg-gray-400 border-3'):
                ui.label('1. Upload Model').classes('text-bold text-2xl')
                ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full bg-gray-300')
        # 2.2 SELECT ACTIVE MODEL CARD
        with ui.row().classes('w-full max-w-6xl justify-center no-wrap'):
            with ui.card().classes('w-1/2 p-4 shadow-lg bg-gray-400 border-3'):
                ui.label('2. Select Active Model').classes('text-bold text-2xl')
                ui.query('body').style('background-color: #EF4444') # Tailwind red-500
                registry_list = ui.list().classes('w-full border rounded mt-2 bg-gray-300')
                model_select = ui.select([], label='Available Models', on_change=lambda e: select_model(e.value)).classes('w-full bg-gray-300 border-2')
                info_label = ui.label('Status: Idle').classes('text-black-800')
                # UN-READY Button
                ui.separator().classes('my-2')
                ui.button('Save Permanently', on_click=save_permanently) \
                    .classes('w-full text-white').props('color=primary')
                ui.button('Un-ready Model', on_click=unready_model) \
                .classes('w-full text-white bg-red-500').props('color=red-200')

    # Call the loader immediately when dashboard is initialized
    load_existing_models()


# Run the application
ui.run(title="ML Model Runner", port=APP_PORT)
