# app.py
import os
import io
import json
import pickle
import sqlite3
import logging
from datetime import datetime
from typing import Optional
from threading import Lock
import re

import pandas as pd
from flask import Flask, request, jsonify, send_file, Blueprint, current_app
from flask_cors import CORS
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher

# -----------------------
# Configure logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Globals & config
# -----------------------
ALLOWED_EXTENSIONS = {'.csv', '.json'}
training_progress = {}
progress_lock = Lock()

# -----------------------
# Helpers
# -----------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    return conn

def init_db(db_path: str):
    """Create DB and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        # workspaces table
        c.execute('''
        CREATE TABLE IF NOT EXISTS workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        # models table -- store training_columns as JSON text
        c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            model_path TEXT NOT NULL,
            target_column TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            accuracy REAL,
            f1_score REAL,
            mse REAL,
            r2_score REAL,
            training_time REAL,
            training_columns TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
        )
        ''')
        # feedback table
        c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            user_id TEXT,
            username TEXT,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
        )
        ''')
        conn.commit()
    finally:
        conn.close()

# -----------------------
# Simple ModelManager
# -----------------------
class ModelManager:
    def __init__(self):
        self._models = {}

    def register_model(self, name: str, model_obj):
        self._models[name] = model_obj

    def get_model(self, name: str):
        return self._models.get(name)

# -----------------------
# Create Flask app
# -----------------------
def create_app():
    app = Flask(__name__)
    CORS(app)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['MODELS_FOLDER'] = os.getenv('MODELS_FOLDER', 'models')
    app.config['DATABASE'] = os.getenv('DATABASE', 'workspace.db')
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload
    app.config['USE_RASA_ONLY'] = os.getenv('USE_RASA_ONLY', 'false').lower() == 'true'
    app.config['RASA_URL'] = os.getenv('RASA_URL', 'http://localhost:5005')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

    # Init DB
    init_db(app.config['DATABASE'])

    # Attach a ModelManager on the app
    app.model_manager = ModelManager()

    # Register blueprint
    app.register_blueprint(api_bp)

    @app.route('/')
    def index():
        return "Backend is working fine!"

    # Debug: list registered routes
    try:
        app.logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            app.logger.info(f"{','.join(sorted(rule.methods))} {rule}")
    except Exception:
        pass

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    return app

# -----------------------
# Blueprint & routes
# -----------------------
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/rasa/parse', methods=['POST'])
def rasa_parse():
    """Proxy to Rasa NLU /model/parse for simple demo/health."""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        text = (payload.get('text') or payload.get('message') or '').strip()
        if not text:
            return jsonify({"error": "Missing text"}), 400
        import requests
        rasa_url = current_app.config.get('RASA_URL', 'http://localhost:5005').rstrip('/') + '/model/parse'
        r = requests.post(rasa_url, json={"text": text}, timeout=5)
        r.raise_for_status()
        data = r.json()
        intent = (data.get('intent') or {}).get('name')
        entities = data.get('entities') or []
        return jsonify({"success": True, "intent": intent, "entities": entities, "raw": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502

@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Workspace endpoints
@api_bp.route('/workspace/create', methods=['POST'])
def create_workspace():
    data = request.get_json(force=True, silent=True) or {}
    workspace_id = data.get('workspace_id')
    user_id = data.get('user_id')
    name = data.get('name')

    if not all([workspace_id, user_id, name]):
        return jsonify({"error": "Missing required fields (workspace_id, user_id, name)"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO workspaces (workspace_id, user_id, name, created_at)
            VALUES (?, ?, ?, ?)
        ''', (workspace_id, user_id, name, datetime.now().isoformat()))
        conn.commit()
        return jsonify({"success": True, "workspace_id": workspace_id})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Workspace already exists"}), 400
    finally:
        conn.close()

@api_bp.route('/workspace/list', methods=['POST'])
def list_workspaces():
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            SELECT workspace_id, name, created_at
            FROM workspaces
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        rows = c.fetchall()
        workspaces = [{"workspace_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]
        return jsonify({"workspaces": workspaces})
    finally:
        conn.close()

@api_bp.route('/workspace/delete', methods=['POST'])
def delete_workspace():
    data = request.get_json(force=True, silent=True) or {}
    workspace_id = data.get('workspace_id')
    user_id = data.get('user_id')

    if not workspace_id or not user_id:
        return jsonify({"error": "Missing required fields (workspace_id, user_id)"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        # Verify ownership
        c.execute('''
            SELECT workspace_id FROM workspaces
            WHERE workspace_id = ? AND user_id = ?
        ''', (workspace_id, user_id))
        if not c.fetchone():
            return jsonify({"error": "Workspace not found or access denied"}), 404

        # Delete associated records
        c.execute('DELETE FROM models WHERE workspace_id = ?', (workspace_id,))
        c.execute('DELETE FROM feedback WHERE workspace_id = ?', (workspace_id,))
        c.execute('DELETE FROM workspaces WHERE workspace_id = ? AND user_id = ?', (workspace_id, user_id))
        conn.commit()

        # Clean up files
        try:
            models_dir = current_app.config['MODELS_FOLDER']
            for f in os.listdir(models_dir):
                if f.startswith(f"{workspace_id}_"):
                    os.remove(os.path.join(models_dir, f))
            uploads_dir = current_app.config['UPLOAD_FOLDER']
            for f in os.listdir(uploads_dir):
                if workspace_id in f:
                    os.remove(os.path.join(uploads_dir, f))
        except Exception as e:
            logger.warning(f"File cleanup failed for workspace {workspace_id}: {e}")

        return jsonify({"success": True, "message": "Workspace deleted successfully"})
    except Exception as e:
        logger.exception("Error deleting workspace")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# Dataset upload & train
@api_bp.route('/dataset/upload', methods=['POST'])
def upload_dataset():
    global training_progress

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    workspace_id = request.form.get('workspace_id')

    # Initialize progress
    with progress_lock:
        training_progress[workspace_id] = {"status": "started", "progress": 5, "message": "Uploading dataset..."}

    try:
        # ---- Save file ----
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        with progress_lock:
            training_progress[workspace_id].update({"progress": 20, "message": "Reading dataset..."})

        # ---- Load dataset ----
        import time
        time.sleep(1)  # simulate
        ext = os.path.splitext(upload_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(upload_path)
        elif ext == '.json':
            try:
                df = pd.read_json(upload_path)
            except ValueError:
                try:
                    df = pd.read_json(upload_path, lines=True)
                except Exception:
                    with open(upload_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
        else:
            raise ValueError('Unsupported file format. Allowed: .csv, .json')

        # Determine target column
        target_column = request.form.get('target_column')
        if not target_column:
            if len(df.columns) == 0:
                raise ValueError('Uploaded dataset has no columns')
            target_column = df.columns[-1]

        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in uploaded dataset columns: {list(df.columns)}"}), 400

        with progress_lock:
            training_progress[workspace_id].update({"progress": 40, "message": "Preprocessing data..."})

        # ---- Train-test split & preprocessing ----
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # One-hot encode categorical/text columns so model.fit receives numeric input
        X_processed = pd.get_dummies(X, drop_first=True)
        training_columns = list(X_processed.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # ---- Training ----
        with progress_lock:
            training_progress[workspace_id].update({"progress": 60, "message": "Training model..."})

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        t0 = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - t0

        try:
            preds = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted'))
        except Exception:
            accuracy = None
            f1 = None

        with progress_lock:
            training_progress[workspace_id].update({"progress": 90, "message": "Saving model..."})

        model_path = os.path.join(current_app.config['MODELS_FOLDER'], f"{workspace_id}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Persist metadata
        try:
            conn = get_db_connection(current_app.config['DATABASE'])
            c = conn.cursor()
            c.execute('''
                INSERT INTO models (workspace_id, model_path, target_column, algorithm, accuracy, f1_score, mse, r2_score, training_time, training_columns, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                workspace_id,
                model_path,
                target_column,
                'RandomForest',
                accuracy,
                f1,
                None,
                None,
                training_time,
                json.dumps(training_columns),
                datetime.now().isoformat()
            ))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # Optionally generate simple Rasa lookup
        try:
            symptom_cols = [c for c in df.columns if c != target_column]
            normalized = []
            for c in symptom_cols:
                s = str(c).strip().lower().replace('_', ' ').replace('-', ' ')
                s = ' '.join(s.split())
                if s and s not in normalized:
                    normalized.append(s)
            lookup_dir = os.path.join(os.getcwd(), 'rasa', 'data', 'lookup')
            os.makedirs(lookup_dir, exist_ok=True)
            lookup_path = os.path.join(lookup_dir, 'symptoms.yml')
            with open(lookup_path, 'w', encoding='utf-8') as f:
                f.write('lookup: symptom\n')
                f.write('examples: |\n')
                for item in normalized:
                    f.write(f"  - {item}\n")
        except Exception:
            pass

        with progress_lock:
            training_progress[workspace_id].update({"progress": 100, "message": "Training complete!", "status": "done"})

        return jsonify({
            "success": True,
            "model_path": model_path,
            "metrics": {"accuracy": accuracy, "f1_score": f1, "training_time": training_time}
        })

    except Exception as e:
        with progress_lock:
            try:
                training_progress[workspace_id].update({"status": "error", "message": str(e)})
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500

@api_bp.route('/progress/<workspace_id>', methods=['GET'])
def get_progress(workspace_id):
    with progress_lock:
        progress_info = training_progress.get(workspace_id, {"status": "unknown", "progress": 0, "message": "No active training"})
    return jsonify(progress_info)

# Predict route
@api_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    workspace_id = request.form.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file format. Allowed: .csv, .json"}), 400

    temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"predict_{workspace_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}")
    try:
        file.save(temp_path)

        conn = get_db_connection(current_app.config['DATABASE'])
        try:
            c = conn.cursor()
            c.execute('''
                SELECT model_path, training_columns, target_column
                FROM models
                WHERE workspace_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (workspace_id,))
            row = c.fetchone()
        finally:
            conn.close()

        if not row:
            return jsonify({"error": "Model not found. Please train a model first."}), 404

        model_path, training_columns_json, target_column = row
        training_columns = json.loads(training_columns_json) if training_columns_json else []

        if not os.path.exists(model_path):
            return jsonify({"error": "Model file missing from disk. Re-train or check server."}), 500

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        ext = os.path.splitext(temp_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(temp_path)
        else:
            try:
                df = pd.read_json(temp_path)
            except ValueError:
                try:
                    df = pd.read_json(temp_path, lines=True)
                except Exception:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)

        if df.empty:
            return jsonify({"error": "Uploaded prediction dataset is empty"}), 400

        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])

        df_processed = pd.get_dummies(df, drop_first=True)
        df_processed = df_processed.reindex(columns=training_columns, fill_value=0)

        try:
            predictions = model.predict(df_processed)
        except Exception as e:
            logger.exception("Model prediction error")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        df_out = df.copy()
        df_out[target_column] = predictions
        cols = [c for c in df_out.columns if c != target_column] + [target_column]
        df_out = df_out[cols]

        preview = df_out.head(100).to_dict('records')

        return jsonify({
            "success": True,
            "predictions": pd.Series(predictions).tolist(),
            "predicted_column": target_column,
            "data_preview": preview,
            "total_rows": len(df_out)
        })
    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# Download predictions
@api_bp.route('/predict/download', methods=['POST'])
def download_predictions():
    data = request.get_json(force=True, silent=True) or {}
    workspace_id = data.get('workspace_id')
    predictions = data.get('predictions')
    original_data = data.get('original_data')

    if not workspace_id or predictions is None:
        return jsonify({"error": "Missing workspace_id or predictions"}), 400

    if not isinstance(original_data, list):
        return jsonify({"error": "original_data must be a list of records"}), 400

    try:
        df = pd.DataFrame(original_data)
        df['predictions'] = predictions

        out_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{workspace_id}_predictions_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
        df.to_csv(out_path, index=False)

        return send_file(out_path, as_attachment=True, download_name='predictions.csv')
    except Exception as e:
        logger.exception("Error in /predict/download")
        return jsonify({"error": str(e)}), 500

# Model info
@api_bp.route('/model/info', methods=['POST'])
def model_info():
    data = request.get_json(force=True, silent=True) or {}
    workspace_id = data.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            SELECT algorithm, accuracy, f1_score, mse, r2_score, training_time, created_at, target_column
            FROM models
            WHERE workspace_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (workspace_id,))
        result = c.fetchone()
    finally:
        conn.close()

    if not result:
        return jsonify({"error": "No model found for this workspace"}), 404

    return jsonify({
        "algorithm": result[0],
        "accuracy": result[1],
        "f1_score": result[2],
        "mse": result[3],
        "r2_score": result[4],
        "training_time": result[5],
        "created_at": result[6],
        "target_column": result[7]
    })

# Training logs
@api_bp.route('/training/logs', methods=['GET'])
def training_logs():
    workspace_id = request.args.get('workspace_id', '')
    level = request.args.get('level', '').upper().strip()

    sample = [
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "INFO", "message": f"Workspace {workspace_id or 'N/A'} training initialized"},
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "INFO", "message": "Loading dataset and preprocessing"},
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "WARN", "message": "Column mismatch detected; applying one-hot alignment"},
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "INFO", "message": "Epoch 1/5 complete; acc=0.84 f1=0.82"},
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "ERROR", "message": "Transient FS latency; retrying"},
        {"ts": datetime.utcnow().isoformat() + 'Z', "level": "INFO", "message": "Training complete; saving model"},
    ]

    if level:
        sample = [l for l in sample if l.get('level') == level]

    return jsonify({"logs": sample}), 200

# Datasets list endpoint
@api_bp.route('/datasets', methods=['GET'])
def list_datasets():
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    try:
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        datasets = []
        for f in os.listdir(uploads_dir):
            if workspace_id in f and (f.endswith('.csv') or f.endswith('.json')):
                file_path = os.path.join(uploads_dir, f)
                stat = os.stat(file_path)
                datasets.append({
                    "id": f,
                    "name": f,
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "url": f"/uploads/{f}"  # Assuming static file serving
                })
        return jsonify({"datasets": datasets})
    except Exception as e:
        logger.exception("Error listing datasets")
        return jsonify({"error": str(e)}), 500

# Models list endpoint
@api_bp.route('/models', methods=['GET'])
def list_models():
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            SELECT id, model_path, target_column, algorithm, accuracy, f1_score, mse, r2_score, training_time, created_at
            FROM models
            WHERE workspace_id = ?
            ORDER BY created_at DESC
        ''', (workspace_id,))
        rows = c.fetchall()
        models = []
        for r in rows:
            models.append({
                "id": r[0],
                "model_path": r[1],
                "target_column": r[2],
                "algorithm": r[3],
                "accuracy": r[4],
                "f1_score": r[5],
                "mse": r[6],
                "r2_score": r[7],
                "training_time": r[8],
                "created_at": r[9]
            })
        return jsonify({"models": models})
    finally:
        conn.close()

# Algorithms endpoint
@api_bp.route('/model/algorithms', methods=['GET'])
def list_algorithms():
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    # For now, return available algorithms. In future, could list used ones from DB
    algorithms = [
        {"name": "RandomForestClassifier", "version": "1.4", "method": "Supervised Learning", "params": {"n_estimators": 100, "random_state": 42}},
        {"name": "LogisticRegression", "version": "1.4", "method": "Supervised Learning", "params": {"C": 1.0, "penalty": "l2"}},
        {"name": "SVM", "version": "1.4", "method": "Supervised Learning", "params": {"C": 1.0, "kernel": "rbf"}},
        {"name": "DecisionTreeClassifier", "version": "1.4", "method": "Supervised Learning", "params": {"max_depth": None, "random_state": 42}}
    ]
    return jsonify({"algorithms": algorithms})

# Feedback endpoints
@api_bp.route('/feedback', methods=['POST'])
def create_feedback():
    data = request.get_json(force=True, silent=True) or {}
    workspace_id = data.get('workspace_id')
    user_id = data.get('user_id')
    username = data.get('username')
    message = data.get('message')
    created_at = data.get('created_at') or datetime.utcnow().isoformat()

    if not workspace_id or not message:
        return jsonify({"error": "Missing required fields (workspace_id, message)"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO feedback (workspace_id, user_id, username, message, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (workspace_id, user_id, username, message, created_at))
        conn.commit()
        return jsonify({"ok": True}), 201
    finally:
        conn.close()

@api_bp.route('/feedback', methods=['GET'])
def list_feedback():
    workspace_id = request.args.get('workspace_id')
    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        if workspace_id:
            c.execute('''
                SELECT username, message, created_at FROM feedback
                WHERE workspace_id = ?
                ORDER BY created_at DESC
            ''', (workspace_id,))
        else:
            c.execute('''
                SELECT username, message, created_at FROM feedback
                ORDER BY created_at DESC
            ''')
        rows = c.fetchall()
        items = [{"username": r[0], "message": r[1], "created_at": r[2]} for r in rows]
        return jsonify({"feedback": items})
    finally:
        conn.close()

# -----------------------
# Chatbot endpoint (complete, fixed)
# -----------------------
@api_bp.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Chatbot with:
    - Small-talk replies for greetings/basic questions
    - Symptom extraction with normalization and fuzzy/synonym matching
    - Optional duration parsing (e.g., 'for 3 days')
    - Prediction using most recent trained model for the workspace
    - Simple RAG over a local factual knowledge base if present
    """
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get('message') or '').strip()
    workspace_id = (data.get('workspace_id') or '').strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Normalize input and remove common filler words
    text = re.sub(
        r"\b(i have|i am having|i feel|my|and|with|for|since|past|having|suffering from|suffering|experiencing|feeling|been|got|from|the|a|an)\b",
        "",
        message.lower()
    )
    text = re.sub(r"[^\w\s,]", " ", text)  # remove punctuation except commas
    text = re.sub(r"\s+", " ", text).strip()

    # Small-talk triggers
    smalltalk_map = {
        r"^(hi|hello|hey|yo|hola|namaste)\b": "Hello! How can I help you today? You can describe your symptoms or ask about your model.",
        r"how are you\??": "I'm operating normally. How are you feeling today?",
        r"who are you\??": "I'm your AI health assistant. Describe symptoms like 'headache and jaw pain for 2 days'.",
        r"help|what can you do": "I can parse your symptoms, match them to your dataset, and predict a likely condition.",
        r"thank(s| you)\b": "You're welcome! Stay healthy.",
        r"bye|goodbye|see you": "Goodbye! Take care and feel better soon.",
    }
    for pattern, resp in smalltalk_map.items():
        if re.search(pattern, text):
            return jsonify({"success": True, "reply": resp})

    try:
        # If using Rasa NLU (optional)
        rasa_entities = []
        if current_app.config.get('USE_RASA_ONLY', False):
            try:
                import requests
                rasa_url = current_app.config.get('RASA_URL', 'http://localhost:5005').rstrip('/') + '/model/parse'
                rr = requests.post(rasa_url, json={"text": message}, timeout=5)
                rr.raise_for_status()
                parsed = rr.json()
                rasa_entities = parsed.get('entities') or []
                intent_name = (parsed.get('intent') or {}).get('name') or ''
                if intent_name in ('greet', 'chitchat', 'smalltalk.greet'):
                    return jsonify({"success": True, "reply": "Hello! How can I help you today?"})
                if intent_name in ('bot_challenge', 'who_are_you'):
                    return jsonify({"success": True, "reply": "I'm your AI health assistant. Describe symptoms like 'headache and jaw pain for 2 days'."})
            except Exception:
                return jsonify({"error": "Rasa NLU is unreachable or failed to parse."}), 502

        # Load latest model metadata
        conn = get_db_connection(current_app.config['DATABASE'])
        c = conn.cursor()
        c.execute('''
            SELECT model_path, training_columns, target_column
            FROM models
            WHERE workspace_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (workspace_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No trained model found for this workspace"}), 404

        model_path, training_columns_json, target_column = row
        training_columns = json.loads(training_columns_json or '[]')

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load dataset file to get raw columns (if available)
        dataset_files = sorted([
            f for f in os.listdir(current_app.config['UPLOAD_FOLDER'])
            if f.endswith('.csv') and workspace_id in f
        ])
        df = None
        if dataset_files:
            try:
                df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], dataset_files[-1]))
            except Exception:
                df = None

        # Use only raw symptom names (avoid one-hot expanded columns)
        if df is not None and target_column in df.columns:
            base_symptoms = [c for c in df.columns if c != target_column]
        else:
            base_symptoms = [c for c in training_columns if '_' not in c]

        base_symptoms_clean = [re.sub(r'[_\-]', ' ', s.lower()).strip() for s in base_symptoms]

        # Helper: singularize simple plurals
        def singularize(word: str) -> str:
            if word.endswith('ies'):
                return word[:-3] + 'y'
            elif word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
            return word

        synonyms = {
            'headache': ['head ache', 'head pain', 'migraine', 'head_ache', 'head-ache', 'headaches'],
            'jaw pain': ['jawpain', 'jaw_pain', 'jaw ache', 'mandible pain'],
            'fever': ['pyrexia', 'high temperature', 'temp'],
            'cough': ['coughing'],
            'sore throat': ['throat pain', 'pharyngitis', 'throat ache'],
        }

        def normalize_symptom(token: str) -> str:
            t = token.strip().lower()
            t = re.sub(r'[_\-]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            t = singularize(t)
            for base, syns in synonyms.items():
                if t == base or t in syns:
                    return base
            return t

        # Duration parsing
        duration_days = None
        m = re.search(r'(?:for|past|since)\s+(\d{1,3})\s*(day|days|d)\b', text)
        if m:
            try:
                duration_days = int(m.group(1))
            except Exception:
                duration_days = None

        # Build candidate symptoms
        candidate_symptoms = []
        if current_app.config.get('USE_RASA_ONLY', False) and rasa_entities:
            for ent in rasa_entities:
                if (ent.get('entity') or '').lower() == 'symptom':
                    val = str(ent.get('value') or '').strip()
                    if val:
                        candidate_symptoms.append(normalize_symptom(val))
        else:
            tokens = re.split(r"[,;\n]|\band\b|\bor\b|\bwith\b|/|\+", text)
            candidate_symptoms = [normalize_symptom(t) for t in tokens if t.strip()]

        # Initialize detected_map using only base_symptoms
        detected_map = {s: 0 for s in base_symptoms}

        def ratio(a, b):
            return SequenceMatcher(None, a, b).ratio()

        # Improved matching: strict threshold + exact/underscore checks
        for cand in candidate_symptoms:
            cand = cand.strip().lower()
            if not cand:
                continue
            cand_norm = re.sub(r'[_\-]', ' ', cand).strip()
            cand_norm = re.sub(r'\s+', ' ', cand_norm)
            for base_raw, base_clean in zip(base_symptoms, base_symptoms_clean):
                score = max(
                    ratio(cand_norm, base_clean),
                    ratio(cand_norm.replace(' ', ''), base_clean.replace(' ', ''))
                )
                if (
                    score >= 0.8
                    or cand_norm == base_clean
                    or cand_norm.replace(' ', '_') == base_clean
                    or cand_norm.replace('_', ' ') == base_clean
                ):
                    detected_map[base_raw] = 1

        # Debug log
        detected_list = [s for s, v in detected_map.items() if v == 1]
        logger.info(f"[Chatbot] Message='{message}' -> Detected: {detected_list}")

        # Build input df aligned with training columns (no new get_dummies)
        user_input_df = pd.DataFrame([detected_map])
        user_input_df = user_input_df.reindex(columns=training_columns, fill_value=0)

        # Prediction with confidence
        try:
            prediction = model.predict(user_input_df)[0]
        except Exception:
            # fallback with safer try
            prediction = model.predict(user_input_df)[0]

        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba_arr = model.predict_proba(user_input_df)
                if hasattr(model, 'classes_') and len(proba_arr) > 0:
                    try:
                        idx = list(model.classes_).index(prediction)
                        proba = float(proba_arr[0][idx])
                    except Exception:
                        proba = float(max(proba_arr[0]))
            except Exception:
                proba = None

        # Build reply text
        if len(detected_list) == 0:
            prefix = "I didn't confidently match specific symptoms, but based on your input, "
        elif len(detected_list) == 1:
            # single symptom prompt
            single_symptom = detected_list[0]
            prompt_more = (
                f"I noticed only one symptom: {single_symptom}. "
                f"Please share if you have any other symptoms (e.g., fever, cough, nausea) for a more accurate assessment."
            )
            return jsonify({
                "success": True,
                "reply": prompt_more,
                "detected_symptoms": detected_list,
                "duration_days": duration_days,
                "need_more_symptoms": True,
                "workspace_id": workspace_id
            })
        else:
            if duration_days is not None:
                prefix = f"I see you have {', '.join(detected_list)} for the past {duration_days} days. "
            else:
                prefix = f"I see you have {', '.join(detected_list)}. "

        rag_suffix = ''
        kb_path = os.path.join(os.getcwd(), 'factual_embeddings.pkl')
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'rb') as f:
                    kb = pickle.load(f)
                if isinstance(kb, dict):
                    facts = []
                    for k in list(kb.keys())[:200]:
                        if isinstance(k, str) and str(prediction).lower() in k.lower():
                            facts.append(k)
                            if len(facts) >= 1:
                                break
                    if facts:
                        rag_suffix = " Note: " + facts[0]
            except Exception:
                pass

        conf_text_line = f"\nConfidence: {proba*100:.1f}%" if proba is not None else ""
        disclaimer_line = "\nPlease consult a medical professional for confirmation."
        reply = f"{prefix}the predicted condition is {prediction}.{rag_suffix}{conf_text_line}{disclaimer_line}"

        return jsonify({
            "success": True,
            "reply": reply,
            "detected_symptoms": detected_list,
            "duration_days": duration_days,
            "confidence": proba,
            "workspace_id": workspace_id
        })

    except Exception as e:
        logger.exception("Error in /chatbot")
        return jsonify({"error": str(e)}), 500

# -----------------------
# Run the app
# -----------------------
if __name__ == '__main__':
    app = create_app()
    # Note: reloader disabled to avoid multiple process issues in some dev environments
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
