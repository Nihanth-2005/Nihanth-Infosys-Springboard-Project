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
import numpy as np
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

def get_all_model_classes():
    """Return all available model classes and their default params."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        'RandomForestClassifier': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
        'LogisticRegression': (LogisticRegression, {'C': 1.0, 'penalty': 'l2', 'random_state': 42, 'max_iter': 1000}),
        'SVM': (SVC, {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}),
        'DecisionTreeClassifier': (DecisionTreeClassifier, {'max_depth': None, 'random_state': 42})
    }

def select_best_model(results):
    """Select the best model based on accuracy."""
    if not results:
        return None

    # Sort by accuracy descending
    sorted_results = sorted(results, key=lambda x: x['accuracy'] or 0, reverse=True)
    return sorted_results[0]

def generate_training_report(workspace_id, df, target_column, algorithm, metrics, training_time, model_path, all_results=None):
    """Generate a PDF training report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet

        report_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{workspace_id}_training_report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf")
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("AI Workspace Training Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Dataset Summary
        story.append(Paragraph("Dataset Summary", styles['Heading2']))
        dataset_data = [
            ["Total Rows", str(len(df))],
            ["Total Columns", str(len(df.columns))],
            ["Target Column", target_column],
            ["Training Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        dataset_table = Table(dataset_data)
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 12))

        # Preprocessing Steps
        story.append(Paragraph("Preprocessing Steps Applied", styles['Heading2']))
        preprocessing_steps = [
            "1. Loaded dataset (CSV/JSON)",
            "2. Selected target column: " + target_column,
            "3. Applied one-hot encoding for categorical features",
            "4. Performed train-test split (80/20)",
            "5. Trained model using " + algorithm
        ]
        for step in preprocessing_steps:
            story.append(Paragraph(step, styles['Normal']))
        story.append(Spacer(1, 12))

        # Model Metrics
        story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
        metrics_data = [["Metric", "Value"]]
        if metrics.get('accuracy') is not None:
            metrics_data.append(["Accuracy", f"{metrics['accuracy']:.4f}"])
        if metrics.get('f1_score') is not None:
            metrics_data.append(["F1 Score", f"{metrics['f1_score']:.4f}"])
        if metrics.get('mse') is not None:
            metrics_data.append(["MSE", f"{metrics['mse']:.4f}"])
        if metrics.get('r2_score') is not None:
            metrics_data.append(["R² Score", f"{metrics['r2_score']:.4f}"])
        metrics_data.append(["Training Time", f"{training_time:.2f} seconds"])

        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 12))

        # Algorithm Comparison (if available)
        if all_results:
            story.append(Paragraph("Algorithm Comparison", styles['Heading2']))
            comparison_data = [["Algorithm", "Accuracy", "F1 Score", "Training Time"]]
            for result in sorted(all_results, key=lambda x: x['accuracy'] or 0, reverse=True):
                comparison_data.append([
                    result['algorithm'],
                    f"{result['accuracy']:.4f}" if result['accuracy'] else "N/A",
                    f"{result['f1_score']:.4f}" if result['f1_score'] else "N/A",
                    f"{result['training_time']:.2f}s"
                ])
            comparison_table = Table(comparison_data)
            comparison_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(comparison_table)
            story.append(Spacer(1, 12))

        # Feature Importance (if available)
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, 'feature_importances_'):
                story.append(Paragraph("Feature Importance", styles['Heading2']))
                # Get training columns from DB
                conn = get_db_connection(current_app.config['DATABASE'])
                c = conn.cursor()
                c.execute('SELECT training_columns FROM models WHERE model_path = ? ORDER BY created_at DESC LIMIT 1', (model_path,))
                row = c.fetchone()
                conn.close()
                if row:
                    training_columns = json.loads(row[0])
                    importances = model.feature_importances_
                    # Sort by importance
                    sorted_idx = importances.argsort()[::-1]
                    importance_data = [["Feature", "Importance"]]
                    for idx in sorted_idx[:10]:  # Top 10
                        importance_data.append([training_columns[idx], f"{importances[idx]:.4f}"])
                    importance_table = Table(importance_data)
                    importance_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(importance_table)
        except Exception:
            pass

        doc.build(story)
        return report_path
    except Exception as e:
        logger.exception("Error generating report")
        return None

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
    algorithm = request.form.get('algorithm', 'RandomForestClassifier')
    automl_mode = request.form.get('automl_mode', 'false').lower() == 'true'

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
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

        if automl_mode:
            # AutoML: Train on all algorithms and select best
            with progress_lock:
                training_progress[workspace_id].update({"progress": 60, "message": "AutoML: Training models with all algorithms..."})

            all_models = get_all_model_classes()
            training_results = []

            total_algorithms = len(all_models)
            progress_per_algorithm = 30 / total_algorithms  # 30% of progress for training

            for i, (alg_name, (model_class, params)) in enumerate(all_models.items()):
                try:
                    model = model_class(**params)
                    t0 = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - t0

                    preds = model.predict(X_test)
                    accuracy = float(accuracy_score(y_test, preds))
                    f1 = float(f1_score(y_test, preds, average='weighted'))
                    mse = float(mean_squared_error(y_test, preds)) if hasattr(y_test, 'dtype') and y_test.dtype.kind in 'fc' else None
                    r2 = float(r2_score(y_test, preds)) if hasattr(y_test, 'dtype') and y_test.dtype.kind in 'fc' else None

                    training_results.append({
                        'algorithm': alg_name,
                        'model': model,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'mse': mse,
                        'r2_score': r2,
                        'training_time': training_time
                    })

                    # Update progress
                    progress = 60 + int((i + 1) * progress_per_algorithm)
                    with progress_lock:
                        training_progress[workspace_id].update({
                            "progress": progress,
                            "message": f"Trained {alg_name} (accuracy: {accuracy:.3f})"
                        })

                except Exception as e:
                    logger.warning(f"Failed to train {alg_name}: {e}")
                    continue

            # Select best model
            best_result = select_best_model(training_results)
            if not best_result:
                raise ValueError("No models could be trained successfully")

            model = best_result['model']
            algorithm = best_result['algorithm']
            accuracy = best_result['accuracy']
            f1 = best_result['f1_score']
            mse = best_result['mse']
            r2 = best_result['r2_score']
            training_time = best_result['training_time']
        else:
            # Manual: Train only selected algorithm
            with progress_lock:
                training_progress[workspace_id].update({"progress": 60, "message": f"Training {algorithm} model..."})

            model_class, params = get_all_model_classes().get(algorithm, (None, None))
            if not model_class:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            model = model_class(**params)
            t0 = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - t0

            preds = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted'))
            mse = float(mean_squared_error(y_test, preds)) if hasattr(y_test, 'dtype') and y_test.dtype.kind in 'fc' else None
            r2 = float(r2_score(y_test, preds)) if hasattr(y_test, 'dtype') and y_test.dtype.kind in 'fc' else None

            training_results = [{
                'algorithm': algorithm,
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'mse': mse,
                'r2_score': r2,
                'training_time': training_time
            }]

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
                algorithm,
                accuracy,
                f1,
                mse,
                r2,
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

        # Generate training report
        with progress_lock:
            training_progress[workspace_id].update({"progress": 95, "message": "Generating report..."})

        report_path = generate_training_report(
            workspace_id, df, target_column, algorithm,
            {"accuracy": accuracy, "f1_score": f1, "mse": mse, "r2_score": r2},
            training_time, model_path, training_results
        )

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
            "report_path": report_path,
            "algorithm": algorithm,
            "metrics": {"accuracy": accuracy, "f1_score": f1, "mse": mse, "r2_score": r2, "training_time": training_time}
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

# Download training report
@api_bp.route('/report/download', methods=['GET'])
def download_report():
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    try:
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        reports = [f for f in os.listdir(uploads_dir)
                   if f.startswith(f"{workspace_id}_training_report_") and f.endswith('.pdf')]

        if not reports:
            return jsonify({"error": "No training report found"}), 404

        # Get latest report
        latest_report = sorted(
            reports,
            key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)),
            reverse=True
        )[0]

        report_path = os.path.join(uploads_dir, latest_report)

        return send_file(report_path, as_attachment=True, download_name='training_report.pdf')

    except Exception as e:
        logger.exception("Error downloading report")
        return jsonify({"error": str(e)}), 500


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

    # Determine health status
    health_status = "unknown"
    if result[1] is not None:  # accuracy
        if result[1] >= 0.85:
            health_status = "good"
        elif result[1] >= 0.60:
            health_status = "moderate"
        else:
            health_status = "poor"

    return jsonify({
        "algorithm": result[0],
        "accuracy": result[1],
        "f1_score": result[2],
        "mse": result[3],
        "r2_score": result[4],
        "training_time": result[5],
        "created_at": result[6],
        "target_column": result[7],
        "health_status": health_status
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

# Dataset insights endpoint
@api_bp.route('/dataset/insights', methods=['GET'])
def dataset_insights():
    import numpy as np
    import pandas as pd
    import json

    def safe(obj):
        """Convert numpy + non-serializable values to safe Python types."""
        if isinstance(obj, (np.int64, np.int32, np.int16, int)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, float)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return obj

    workspace_id = request.args.get('workspace_id')
    dataset_id = request.args.get('dataset_id')

    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    try:
        uploads_dir = current_app.config['UPLOAD_FOLDER']
        datasets = [
            f for f in os.listdir(uploads_dir)
            if f.endswith(('.csv', '.json')) and workspace_id in f
        ]

        if not datasets:
            return jsonify({"error": "No datasets found for this workspace"}), 404

        # Select file
        if dataset_id and dataset_id in datasets:
            dataset_file = dataset_id
        else:
            dataset_file = sorted(
                datasets,
                key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)),
                reverse=True
            )[0]

        file_path = os.path.join(uploads_dir, dataset_file)
        ext = os.path.splitext(dataset_file)[1].lower()

        # Load dataset
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_json(file_path)
        except Exception:
            # For JSON-lines
            df = pd.read_json(file_path, lines=True)

        if df.empty:
            return jsonify({"error": "Dataset is empty"}), 400

        # Column types
        column_types = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype == "object":
                unique_ratio = df[col].nunique() / len(df)
                column_types[col] = "categorical" if unique_ratio < 0.1 else "text"
            elif "int" in dtype:
                column_types[col] = "integer"
            elif "float" in dtype:
                column_types[col] = "float"
            else:
                column_types[col] = dtype

        # Unique values
        unique_values = {}
        for col in df.columns:
            try:
                unique_values[col] = {
                    safe(k): safe(v)
                    for k, v in df[col].value_counts().head(10).to_dict().items()
                }
            except:
                unique_values[col] = {}

        # Correlation
        numeric_cols = df.select_dtypes(include=[np.number])
        correlation = (
            numeric_cols.corr().applymap(safe).to_dict()
            if numeric_cols.shape[1] > 1
            else {}
        )

        # Class distribution (target = low cardinality last column)
        class_distribution = {}
        potential_targets = [c for c in df.columns if df[c].nunique() <= 20]

        if potential_targets:
            target = potential_targets[-1]

            class_distribution = {
                safe(k): safe(v)
                for k, v in df[target].value_counts().to_dict().items()
            }

        # Sample rows
        sample_rows = {
            "top_5": df.head(5).applymap(safe).to_dict(orient="records"),
            "bottom_5": df.tail(5).applymap(safe).to_dict(orient="records"),
        }

        insights = {
            "dataset_id": dataset_file,
            "total_rows": safe(len(df)),
            "total_columns": safe(len(df.columns)),
            "column_types": column_types,
            "unique_values": unique_values,
            "correlation": correlation,
            "class_distribution": class_distribution,
            "sample_rows": sample_rows,
        }

        # FINAL FIX — convert everything to JSON-safe types
        insights = json.loads(json.dumps(insights, default=safe))

        return jsonify({"insights": insights})

    except Exception as e:
        logger.exception("Error in dataset insights")
        return jsonify({"error": str(e)}), 500

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

# Model comparison endpoint
@api_bp.route('/model/comparison', methods=['GET'])
def model_comparison():
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({"error": "Missing workspace_id"}), 400

    conn = get_db_connection(current_app.config['DATABASE'])
    try:
        c = conn.cursor()
        c.execute('''
            SELECT id, algorithm, accuracy, f1_score, mse, r2_score, training_time, created_at, target_column
            FROM models
            WHERE workspace_id = ?
            ORDER BY created_at DESC
        ''', (workspace_id,))
        rows = c.fetchall()
        models = []
        for r in rows:
            health_status = "unknown"
            if r[2] is not None:  # accuracy
                if r[2] >= 0.85:
                    health_status = "good"
                elif r[2] >= 0.60:
                    health_status = "moderate"
                else:
                    health_status = "poor"
            models.append({
                "id": r[0],
                "algorithm": r[1],
                "accuracy": r[2],
                "f1_score": r[3],
                "mse": r[4],
                "r2_score": r[5],
                "training_time": r[6],
                "created_at": r[7],
                "target_column": r[8],
                "health_status": health_status
            })
        return jsonify({"models": models})
    finally:
        conn.close()

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
            base_symptoms = training_columns

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
