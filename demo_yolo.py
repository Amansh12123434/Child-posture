# drone_demo_distress_dashboard_v2.py
"""
Enhanced Missing Child / Distress Detection Dashboard - Ready to run

Features:
 - YOLOv8 person detection (GPU if available)
 - MediaPipe pose + hands
 - Optional DeepFace (gender/age/emotion) - best-effort, optional
 - MJPEG live stream at /video_feed (used by dashboard)
 - Snapshots saved to SNAPSHOT_DIR
 - /events -> JSON log of snapshot events (used by Recorded Data)
 - /stats  -> live counts + drone health + model summary for dashboard polling

Edit CONFIG section for file paths and model path.
"""

import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, send_from_directory, url_for

# optional heavy dependencies
try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("mediapipe is required. Install with `pip install mediapipe`") from e

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# Optional: DeepFace for gender/age/emotion (best-effort)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

# ---------------- CONFIG ----------------
INPUT_PATH = r"D:\child detections\Child-posture\123.mp4"   # change as needed
SNAPSHOT_DIR = r"D:\child detections\Child-posture\snapshots"
LOG_JSON = "missing_log.json"
CHILD_RATIO = 0.40            # bbox height < CHILD_RATIO*frame_height => child (heuristic)
MISSING_FRAMES = 30
YOLO_MODEL_PATH = "yolov8s.pt"  # <-- user requested "yolo model" -> using yolov8s.pt
USE_DEVICE = 0                 # GPU device id (0) or "cpu" to force CPU
# MediaPipe settings
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# create snapshot dir if missing
Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ---------------- Globals ----------------
latest_frame_jpeg = None   # last encoded MJPEG frame bytes
processing_thread = None
processing_running = False

events_lock = threading.Lock()
stats_lock = threading.Lock()
reported_ids = set()

# live stats returned by /stats
live_stats = {
    "maleCount": 0,
    "femaleCount": 0,
    "childCount": 0,
    "objectCount": 0,
    "droneHealth": {"battery": 100.0, "gps": "Active", "latency_ms": 0},
    "modelName": YOLO_MODEL_PATH,
    "modelSummary": {"persons": 0, "children": 0, "distress": 0}
}

# ---------------- Simple Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=MISSING_FRAMES):
        self.next_id = 0
        self.objects = {}      # oid -> (cx, cy)
        self.bboxes = {}       # oid -> (x,y,w,h)
        self.disappeared = {}  # oid -> count
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        oid = self.next_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_id += 1
        return oid

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.bboxes.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    remove.append(oid)
            for oid in remove:
                self.deregister(oid)
            return self.objects, self.bboxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w/2), int(y + h/2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]), rects[i])
            return self.objects, self.bboxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        try:
            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)
        except Exception:
            # fallback: re-register
            for col in range(len(rects)):
                self.register(tuple(input_centroids[col]), rects[col])
            return self.objects, self.bboxes

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self.objects[oid] = tuple(input_centroids[col])
            self.bboxes[oid] = rects[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        unused_cols = set(range(0, len(rects))) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]), rects[col])

        return self.objects, self.bboxes

# ---------------- Utilities ----------------
def safe_write_json(event):
    """Append event (dict) to JSON list file in a thread-safe manner"""
    with events_lock:
        data = []
        if os.path.exists(LOG_JSON):
            try:
                with open(LOG_JSON, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(event)
        with open(LOG_JSON, "w") as f:
            json.dump(data, f, indent=2)

def make_snapshot_filename(prefix, oid, frame_no):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_id{oid}_frame{frame_no}_{ts}.jpg"

# ---------------- Processing Thread ----------------
def processing_worker():
    """Background worker: reads video, runs detections, annotates frames,
       saves snapshots & logs events, updates latest_frame_jpeg for streaming."""
    global latest_frame_jpeg, processing_running, reported_ids, live_stats

    # load YOLO if available
    yolo_model = None
    if ULTRALYTICS_AVAILABLE:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[INFO] Loaded YOLO model: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Failed to load YOLO model '{YOLO_MODEL_PATH}': {e}. HOG fallback will be used.")
            yolo_model = None
    else:
        print("[WARN] ultralytics not available - HOG will be used.")

    # HOG fallback
    hog = None
    if yolo_model is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # face detector (Haar) for "face found" fallback
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # mediapipe detectors
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands_detector = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                                    min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)
    pose_detector = mp_pose.Pose(static_image_mode=False,
                                min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                                min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE)

    # video capture
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"[processing_worker] ERROR: cannot open video: {INPUT_PATH}")
        processing_running = False
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    tracker = CentroidTracker(max_disappeared=MISSING_FRAMES)
    reported_ids_local = set()   # per-run reported ids

    frame_no = 0
    print("[processing_worker] Started processing.")

    try:
        while processing_running:
            ret, frame = cap.read()
            if not ret:
                print("[processing_worker] End of video reached or cannot read frame.")
                break
            frame_no += 1

            # --- DETECTION: people (YOLO preferred, HOG fallback) ---
            rects = []
            if yolo_model is not None:
                try:
                    results = yolo_model.predict(frame, device=USE_DEVICE, verbose=False)
                    if len(results) and hasattr(results[0], 'boxes'):
                        for box in results[0].boxes.xyxy:  # xyxy (float32 tensor)
                            x1, y1, x2, y2 = map(int, box)
                            rects.append((x1, y1, x2 - x1, y2 - y1))
                except Exception as e:
                    print("[WARN] YOLO predict failed:", e)
                    rects = []
            if (yolo_model is None) or (len(rects) == 0 and hog is not None):
                found, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
                rects = [tuple(xywh) for xywh in found] if len(found) else []

            # update tracker
            objects, bboxes = tracker.update(rects)

            # face detection (Haar) for "face_found" fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            face_centers = [(int(x+w/2), int(y+h/2)) for (x,y,w,h) in faces]

            # pose detection for shoulder/hip positions
            pose_results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            left_shoulder_y = None
            right_shoulder_y = None
            hip_y = None
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                left_shoulder_y = lm[11].y * height if len(lm) > 11 else None
                right_shoulder_y = lm[12].y * height if len(lm) > 12 else None
                hip_y_vals = []
                if len(lm) > 23: hip_y_vals.append(lm[23].y * height)
                if len(lm) > 24: hip_y_vals.append(lm[24].y * height)
                if hip_y_vals: hip_y = float(np.mean(hip_y_vals))

            # Per-frame counters (to publish to /stats)
            male_ct = 0
            female_ct = 0
            child_ct = 0
            distress_ct = 0

            # Process each detected/tracked bbox
            for oid in list(bboxes.keys()):
                bbox = bboxes.get(oid)
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                # clamp
                x = max(0, int(x)); y = max(0, int(y))
                w = max(1, int(w)); h = max(1, int(h))
                if x + w > width: w = width - x
                if y + h > height: h = height - y

                # heuristics
                is_child = (h < int(height * CHILD_RATIO))
                face_found = any((x <= fx <= x + w) and (y <= fy <= y + h) for (fx, fy) in face_centers)

                # crop person region for hands/face/DeepFace
                crop = frame[y:y+h, x:x+w]
                raised_hand = False
                # MEDIA PIPE HANDS
                if crop.size > 0:
                    try:
                        hands_res = hands_detector.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if hands_res.multi_hand_landmarks:
                            for hand_landmarks in hands_res.multi_hand_landmarks:
                                wrist = hand_landmarks.landmark[0]
                                wrist_y_full = y + int(wrist.y * h)
                                shoulder_y = None
                                if left_shoulder_y is not None and right_shoulder_y is not None:
                                    shoulder_y = min(left_shoulder_y, right_shoulder_y)
                                elif left_shoulder_y is not None:
                                    shoulder_y = left_shoulder_y
                                elif right_shoulder_y is not None:
                                    shoulder_y = right_shoulder_y
                                if shoulder_y is not None:
                                    if wrist_y_full < (shoulder_y - 10):
                                        raised_hand = True
                                        break
                                else:
                                    bbox_center_y = y + h/2
                                    if wrist_y_full < bbox_center_y:
                                        raised_hand = True
                                        break
                    except Exception:
                        pass

                # optional DeepFace age/gender/emotion - best-effort (slow)
                distress_emotion = False
                if DEEPFACE_AVAILABLE and crop.size > 0:
                    try:
                        analysis = DeepFace.analyze(crop, actions=['age','gender','emotion'], enforce_detection=False)
                        gender = str(analysis.get('gender','')).lower()
                        age_val = int(round(analysis.get('age', 0) or 0))
                        dominant_emotion = analysis.get('dominant_emotion', '').lower()
                        if gender.startswith('m'): male_ct += 1
                        elif gender.startswith('f'): female_ct += 1
                        if age_val and age_val < 16:
                            is_child = True
                        if dominant_emotion in ['fear', 'sad']:
                            distress_emotion = True
                    except Exception:
                        pass

                if is_child:
                    child_ct += 1

                # additional distress heuristic: fall / head below hip (if pose available)
                fall_detected = False
                if pose_results.pose_landmarks and crop.size > 0:
                    try:
                        nose_y = pose_results.pose_landmarks.landmark[0].y * height if len(pose_results.pose_landmarks.landmark) > 0 else None
                        if nose_y and hip_y and (nose_y > hip_y + 30):
                            fall_detected = True
                    except Exception:
                        pass

                # final distress condition (child + raised hand + no face) OR emotion OR fall
                distress = (is_child and raised_hand and (not face_found)) or distress_emotion or fall_detected

                if distress:
                    distress_ct += 1

                if distress and oid not in reported_ids:
                    # save snapshot & log
                    snap_name = make_snapshot_filename("distress", oid, frame_no)
                    snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)  # save full frame for context
                    event = {
                        "id": int(oid),
                        "frame": int(frame_no),
                        "timestamp": time.time(),
                        "snapshot": snap_name,
                        "kind": "distress"
                    }
                    safe_write_json(event)
                    reported_ids.add(oid)
                    print(f"[SNAPSHOT] saved: {snap_path}")

                # draw bounding box and label
                color = (0, 0, 255) if distress else (0, 255, 0)
                label = f"ID {oid}"
                if is_child: label += " Child"
                if raised_hand: label += " RaisedHand"
                if fall_detected: label += " Fall"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Missing-child heuristic: check IDs that disappeared recently (mark missing)
            try:
                all_ids_range = range(0, tracker.next_id)
                for oid_check in all_ids_range:
                    if oid_check not in tracker.bboxes and oid_check not in reported_ids:
                        snap_name = make_snapshot_filename("missing", oid_check, frame_no)
                        snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                        cv2.imwrite(snap_path, frame)
                        event = {
                            "id": int(oid_check),
                            "frame": int(frame_no),
                            "timestamp": time.time(),
                            "snapshot": snap_name,
                            "kind": "missing"
                        }
                        safe_write_json(event)
                        reported_ids.add(oid_check)
                        print(f"[SNAPSHOT] saved (missing): {snap_path}")
            except Exception:
                pass

            # Update live_stats (thread-safe)
            with stats_lock:
                live_stats["maleCount"] = int(male_ct)
                live_stats["femaleCount"] = int(female_ct)
                live_stats["childCount"] = int(child_ct)
                live_stats["objectCount"] = int(len(bboxes))
                live_stats["modelSummary"]["persons"] = int(len(bboxes))
                live_stats["modelSummary"]["children"] = int(child_ct)
                # cumulative distress in this frame
                live_stats["modelSummary"]["distress"] = int(distress_ct)
                # simple droneHealth simulation (you can hook real telemetry here)
                live_stats["droneHealth"]["battery"] = max(5.0, live_stats["droneHealth"]["battery"] - 0.02)
                live_stats["droneHealth"]["gps"] = "Active"
                live_stats["droneHealth"]["latency_ms"] = int(np.clip(np.random.normal(loc=45, scale=10), 10, 250))

            # encode frame for MJPEG streaming (for /video_feed)
            try:
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                latest_frame_jpeg = jpeg.tobytes()
            except Exception:
                latest_frame_jpeg = None

            # sleep to match video fps
            if fps > 0:
                time.sleep(max(0.001, 1.0 / fps))

    finally:
        try:
            hands_detector.close()
            pose_detector.close()
        except Exception:
            pass
        cap.release()
        processing_running = False
        print("[processing_worker] Stopped processing.")

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config['SNAPSHOT_FOLDER'] = SNAPSHOT_DIR

# ---------------- Dashboard HTML (keeps your styling exactly)
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drone Monitoring Dashboard - Full</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-dark: #0f172a;
            --secondary-dark: #1e293b;
            --accent-blue: #0ea5e9;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-orange: #f59e0b;
        }

        body {
            background: linear-gradient(135deg, var(--primary-dark) 0%, #1e1b4b 100%);
            color: #e2e8f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }

        .dashboard-container {
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            padding: 20px;
        }

        .header-gradient {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        .metric-card {
            background: var(--secondary-dark);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .live-indicator {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .tab-active {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%) !important;
            color: white !important;
            border: none !important;
        }

        .video-feed {
            background: linear-gradient(45deg, #1e293b, #334155);
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            position: relative;
            min-height: 250px;
        }

        .video-feed::after {
            content: 'LIVE FEED';
            position: absolute;
            top: 10px;
            right: 10px;
            background: #ef4444;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }

        .chart-container {
            background: var(--secondary-dark);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-online { background: var(--accent-green); }
        .status-warning { background: var(--accent-orange); }
        .status-offline { background: #ef4444; }

        .progress-bar-custom {
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            border-radius: 10px;
        }

        .data-flow {
            background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
            animation: flow 3s linear infinite;
        }

        @keyframes flow {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .glow-effect {
            box-shadow: 0 0 20px rgba(14, 165, 233, 0.3);
        }

        .controls { margin-bottom: 12px; }

        .snapshot-thumb { width: 80px; height: auto; border-radius:6px; cursor:pointer; border:1px solid rgba(255,255,255,0.08); }
    </style>
</head>
<body class="py-4">
<div class="container dashboard-container">
    <!-- Header -->
    <div class="row py-2 border-bottom border-secondary">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="header-gradient display-6 fw-bold">
                        <i class="fas fa-satellite me-3"></i>
                        AI-Driven Multi-Drone Monitoring Dashboard
                    </h1>
                    <p class="text-muted mb-0">Real-time surveillance and object detection platform</p>
                </div>
                <div class="text-end">
                    <span class="badge bg-danger live-indicator me-2">LIVE</span>
                    <small class="text-muted">Last update: <span id="current-time">Just now</span></small>
                </div>
            </div>
        </div>
    </div>

    <!-- Controls -->
    <div class="row controls">
        <div class="col">
            <button id="startBtn" class="btn btn-success btn-sm me-2">Start Processing</button>
            <button id="stopBtn" class="btn btn-danger btn-sm">Stop Processing</button>
        </div>
    </div>

    <!-- Quick Stats -->
    <div class="row py-3">
        <div class="col-md-3 mb-3">
            <div class="metric-card p-3 text-center">
                <i class="fas fa-male fa-2x text-info mb-2"></i>
                <h4 class="mb-1" id="maleCount">0</h4>
                <small class="text-muted">Male Count</small>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="metric-card p-3 text-center">
                <i class="fas fa-female fa-2x text-warning mb-2"></i>
                <h4 class="mb-1" id="femaleCount">0</h4>
                <small class="text-muted">Female Count</small>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="metric-card p-3 text-center">
                <i class="fas fa-child fa-2x text-success mb-2"></i>
                <h4 class="mb-1" id="childCount">0</h4>
                <small class="text-muted">Child Count</small>
            </div>
        </div>
        <div class="col-md-3 mb-3">
            <div class="metric-card p-3 text-center">
                <i class="fas fa-box fa-2x text-primary mb-2"></i>
                <h4 class="mb-1" id="objectCount">0</h4>
                <small class="text-muted">Objects Tracked</small>
            </div>
        </div>
    </div>

    <!-- Main Content Tabs -->
    <div class="row">
        <div class="col-12">
            <ul class="nav nav-pills mb-4" id="dashboardTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active tab-active" id="tab1-tab" data-bs-toggle="pill" data-bs-target="#tab1" type="button">
                        <i class="fas fa-play-circle me-2"></i>Live Monitoring
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab2-tab" data-bs-toggle="pill" data-bs-target="#tab2" type="button">
                        <i class="fas fa-search me-2"></i>Object Surveillance
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab3-tab" data-bs-toggle="pill" data-bs-target="#tab3" type="button">
                        <i class="fas fa-heartbeat me-2"></i>Drone Health
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="tab4-tab" data-bs-toggle="pill" data-bs-target="#tab4" type="button">
                        <i class="fas fa-database me-2"></i>Recorded Data
                    </button>
                </li>
            </ul>

            <div class="tab-content" id="dashboardTabsContent">
                <!-- Tab 1: Live Monitoring -->
                <div class="tab-pane fade show active" id="tab1" role="tabpanel">
                    <div class="row">
                        <!-- Drone 1 Feed -->
                        <div class="col-md-6 mb-4">
                            <div class="video-feed p-3">
                                <h6><i class="fas fa-drone me-2"></i>Drone 1 - Live Feed <span class="status-indicator status-online"></span></h6>
                                <div class="bg-dark rounded p-2 text-center">
                                    <!-- live MJPEG stream -->
                                    <img id="drone1Feed" src="{{ url_for('video_feed') }}" style="width:100%; border-radius:12px; border:2px solid rgba(255,255,255,0.1);" alt="Live Video">
                                    <p class="text-muted mt-2 mb-0">Live video stream with object detection overlay</p>
                                </div>
                            </div>
                        </div>

                        <!-- Model Summary (replaces Drone 2) -->
                        <div class="col-md-6 mb-4">
                            <div class="video-feed p-3">
                                <h6><i class="fas fa-robot me-2"></i>AI Model Summary <span class="status-indicator status-online"></span></h6>
                                <div class="bg-dark rounded p-4 text-start">
                                    <p class="mb-1"><strong>Model:</strong> <span id="modelName">--</span></p>
                                    <p class="mb-1"><strong>Persons (frame):</strong> <span id="modelPersons">0</span></p>
                                    <p class="mb-1"><strong>Children (frame):</strong> <span id="modelChildren">0</span></p>
                                    <p class="mb-1"><strong>Distress alerts (frame):</strong> <span id="modelDistress">0</span></p>
                                    <p class="text-muted mt-2 mb-0"><em>Comments:</em> Model is used to detect people; for better child/adult separation train a dedicated classifier (Roboflow/YOLO).</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Charts -->
                    <div class="row">
                        <div class="col-md-6 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-chart-pie me-2"></i>Gender Distribution</h6>
                                <div class="bg-dark rounded p-4 text-center">
                                    <i class="fas fa-chart-pie fa-3x text-info mb-3"></i>
                                    <p class="text-muted">Pie chart showing male/female/child distribution</p>
                                    <div class="mt-3">
                                        <span class="badge bg-info me-2" id="pieMale">Male: 0</span>
                                        <span class="badge bg-warning me-2" id="pieFemale">Female: 0</span>
                                        <span class="badge bg-success me-2" id="pieChild">Child: 0</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-chart-line me-2"></i>Count Over Time</h6>
                                <div class="bg-dark rounded p-4 text-center">
                                    <i class="fas fa-chart-line fa-3x text-success mb-3"></i>
                                    <p class="text-muted">Line chart showing detection trends over time</p>
                                    <div class="mt-3">
                                        <span class="badge bg-info me-2">Peak: 0</span>
                                        <span class="badge bg-warning me-2">Avg: 0</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Object Surveillance -->
                <div class="tab-pane fade" id="tab2" role="tabpanel">
                    <div class="row">
                        <div class="col-md-8 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-map me-2"></i>Detection Map</h6>
                                <div class="bg-dark rounded p-4 text-center" style="min-height: 300px;">
                                    <i class="fas fa-map-marked-alt fa-3x text-warning mb-3"></i>
                                    <p class="text-muted">Interactive map showing object locations</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-list me-2"></i>Detected Objects</h6>
                                <div class="table-responsive">
                                    <table class="table table-dark table-sm">
                                        <thead>
                                            <tr>
                                                <th>Type</th>
                                                <th>Count</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td><i class="fas fa-user text-info me-2"></i>Person</td>
                                                <td id="tblPerson">0</td>
                                                <td>--</td>
                                            </tr>
                                            <tr>
                                                <td><i class="fas fa-car text-warning me-2"></i>Vehicle</td>
                                                <td id="tblVehicle">0</td>
                                                <td>--</td>
                                            </tr>
                                            <tr>
                                                <td><i class="fas fa-paw text-success me-2"></i>Animal</td>
                                                <td id="tblAnimal">0</td>
                                                <td>--</td>
                                            </tr>
                                            <tr>
                                                <td><i class="fas fa-box text-primary me-2"></i>Other Objects</td>
                                                <td id="tblOther">0</td>
                                                <td>--</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab 3: Drone Health & Modeling -->
                <div class="tab-pane fade" id="tab3" role="tabpanel">
                    <div class="row">
                        <div class="col-md-6 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-heartbeat me-2"></i>Drone Health Status</h6>
                                <div class="p-3 bg-dark rounded">
                                    <p>Battery Level: <span id="battery">100</span>%</p>
                                    <p>GPS Lock: <span id="gpsStatus">Active</span></p>
                                    <p>Latency: <span id="latency">0</span> ms</p>
                                    <p>Motor Status: <span class="fw-bold">All Functional</span></p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6 mb-4">
                            <div class="chart-container">
                                <h6 class="mb-3"><i class="fas fa-project-diagram me-2"></i>Modeling & Analysis</h6>
                                <div class="p-3 bg-dark rounded">
                                    <p>Flight Simulation Accuracy: <span class="fw-bold">--</span></p>
                                    <p>Object Tracking Precision: <span class="fw-bold">--</span></p>
                                    <p>Data Stream Latency: <span class="fw-bold">--</span></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Tab 4: Recorded Data -->
                <div class="tab-pane fade" id="tab4" role="tabpanel">
                    <div class="chart-container">
                        <h6 class="mb-3"><i class="fas fa-database me-2"></i>Recorded Data Logs</h6>
                        <div class="table-responsive">
                            <table class="table table-dark table-striped table-sm">
                                <thead>
                                    <tr>
                                        <th>Timestamp</th>
                                        <th>Type</th>
                                        <th>Snapshot</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody id="recordedLogs">
                                    <!-- filled by JS -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <div class="row mt-4 border-top border-secondary pt-3">
        <div class="col-12 text-center">
            <small class="text-muted">© 2025 Drone AI Monitoring Dashboard. All rights reserved.</small>
        </div>
    </div>
</div>

<!-- Scripts -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // Update current time every second
    function updateTime() {
        const now = new Date();
        const formatted = now.getFullYear() + '-' +
            String(now.getMonth()+1).padStart(2,'0') + '-' +
            String(now.getDate()).padStart(2,'0') + ' ' +
            String(now.getHours()).padStart(2,'0') + ':' +
            String(now.getMinutes()).padStart(2,'0') + ':' +
            String(now.getSeconds()).padStart(2,'0');
        document.getElementById('current-time').innerText = formatted;
    }
    setInterval(updateTime, 1000);
    updateTime();

    // Poll /stats to update counts and drone health
    async function pollStats(){
      try {
        const res = await fetch('/stats');
        if(!res.ok) return;
        const data = await res.json();
        if(document.getElementById('maleCount')) document.getElementById('maleCount').innerText = data.maleCount ?? 0;
        if(document.getElementById('femaleCount')) document.getElementById('femaleCount').innerText = data.femaleCount ?? 0;
        if(document.getElementById('childCount')) document.getElementById('childCount').innerText = data.childCount ?? 0;
        if(document.getElementById('objectCount')) document.getElementById('objectCount').innerText = data.objectCount ?? 0;
        // pie badges
        if(document.getElementById('pieMale')) document.getElementById('pieMale').innerText = 'Male: ' + (data.maleCount ?? 0);
        if(document.getElementById('pieFemale')) document.getElementById('pieFemale').innerText = 'Female: ' + (data.femaleCount ?? 0);
        if(document.getElementById('pieChild')) document.getElementById('pieChild').innerText = 'Child: ' + (data.childCount ?? 0);

        // drone health
        if(document.getElementById('battery')) document.getElementById('battery').innerText = Math.round(data.droneHealth.battery);
        if(document.getElementById('gpsStatus')) document.getElementById('gpsStatus').innerText = data.droneHealth.gps;
        if(document.getElementById('latency')) document.getElementById('latency').innerText = data.droneHealth.latency_ms;

        // model summary
        if(document.getElementById('modelName')) document.getElementById('modelName').innerText = data.modelName ?? '--';
        if(document.getElementById('modelPersons')) document.getElementById('modelPersons').innerText = data.modelSummary?.persons ?? 0;
        if(document.getElementById('modelChildren')) document.getElementById('modelChildren').innerText = data.modelSummary?.children ?? 0;
        if(document.getElementById('modelDistress')) document.getElementById('modelDistress').innerText = data.modelSummary?.distress ?? 0;
      } catch (e) {
        console.log('stats poll error', e);
      }
    }
    setInterval(pollStats, 2000);
    pollStats();

    // Start / Stop processing
    document.getElementById('startBtn').addEventListener('click', async ()=>{
        await fetch('/start');
        setTimeout(pollStats, 1000);
        setTimeout(loadEvents, 1000);
    });
    document.getElementById('stopBtn').addEventListener('click', async ()=>{
        await fetch('/stop');
    });

    // load events into Recorded Data
    async function loadEvents(){
      try {
        const res = await fetch('/events');
        if(!res.ok) return;
        const data = await res.json();
        const tbody = document.getElementById('recordedLogs');
        tbody.innerHTML = '';
        data.slice().reverse().forEach(e=>{
            const tr = document.createElement('tr');
            const dt = new Date(e.timestamp*1000);
            const timeStr = dt.getFullYear() + '-' + String(dt.getMonth()+1).padStart(2,'0') + '-' + String(dt.getDate()).padStart(2,'0') + ' ' + dt.toLocaleTimeString();
            const thumb = document.createElement('img');
            thumb.src = '/snapshots/' + encodeURIComponent(e.snapshot);
            thumb.className = 'snapshot-thumb';
            thumb.onclick = ()=> window.open(thumb.src, '_blank');
            tr.innerHTML = `<td>${timeStr}</td>
                            <td>${e.kind}</td>
                            <td></td>
                            <td>id:${e.id} frame:${e.frame}</td>`;
            tr.cells[2].appendChild(thumb);
            tbody.appendChild(tr);
        });
      } catch (e) {
        console.log('events load error', e);
      }
    }
    setInterval(loadEvents, 5000);
    loadEvents();
</script>
</body>
</html>
"""

# ---------------- Flask Endpoints ----------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

def mjpeg_generator():
    """Yields MJPEG frames from latest_frame_jpeg global."""
    global latest_frame_jpeg
    while True:
        frame = None
        try:
            frame = latest_frame_jpeg
        except Exception:
            frame = None
        if frame is None:
            blank = np.zeros((240,320,3), dtype=np.uint8)
            _, jpeg = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame = jpeg.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/events")
def events():
    if os.path.exists(LOG_JSON):
        try:
            with open(LOG_JSON, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    else:
        data = []
    for e in data:
        e["snapshot_url"] = url_for("snapshot_file", filename=e["snapshot"])
    return jsonify(data)

@app.route("/stats")
def stats():
    with stats_lock:
        return jsonify(live_stats)

@app.route("/snapshots/<path:filename>")
def snapshot_file(filename):
    return send_from_directory(app.config['SNAPSHOT_FOLDER'], filename)

@app.route("/start")
def start_processing():
    global processing_thread, processing_running, reported_ids
    with threading.Lock():
        if processing_running:
            return jsonify({"status":"already_running"})
        processing_running = True
        reported_ids = set()
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()
        return jsonify({"status":"started"})

@app.route("/stop")
def stop_processing():
    global processing_running
    with threading.Lock():
        if not processing_running:
            return jsonify({"status":"not_running"})
        processing_running = False
        return jsonify({"status":"stopping"})

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Starting app. Open http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
