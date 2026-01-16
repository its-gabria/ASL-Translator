import re
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from flask import Flask, Response
from flask_cors import CORS

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Setup TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Webcam
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

sentence = ""
current_word = ""
last_letter_time = time.time()

last_prediction = None
cooldown = 1.5  # seconds before same gesture can be accepted again

# Special gestures mapping
special_gestures = {
    "HELLO": "hello",
    "ILOVEYOU": "I love you",
    "THANKS": "thank you"
}

latest_prediction = ""

def normalize_raw(raw):
    """Turn model output into a str (handles bytes/numpy types)."""
    if isinstance(raw, bytes):
        try:
            return raw.decode('utf-8', errors='ignore')
        except Exception:
            return str(raw)
    return str(raw)

def clean_prediction(pred):
    """
    Clean model prediction:
    - Remove trailing 'LEFT' or 'RIGHT' (case-insensitive), with or without separator
    - Normalize to uppercase
    """
    pred = normalize_raw(pred).strip()

    # Handle cases like HRight, GLeft, ORight, A_LEFT, A-Right, A Right
    pred = re.sub(r'[_\-\s]?(left|right)$', '', pred, flags=re.IGNORECASE)

    pred = pred.strip()
    if pred == "":
        return ""
    return pred.upper()

def generate_frames():
    global sentence, current_word, last_letter_time, last_prediction, latest_prediction
    while True:
        data_aux, x_, y_ = [], [], []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect normalized coordinates
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            # Raw prediction from model
            prediction = model.predict([np.asarray(data_aux)])
            raw_pred = prediction[0]
            clean_pred = clean_prediction(raw_pred)  # cleaned & uppercased

            if clean_pred == "":
                last_prediction = None
                cv2.putText(frame, f"RAW:{normalize_raw(raw_pred)}", (10, H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # Accept only if new or cooldown passed
                if clean_pred != last_prediction or (time.time() - last_letter_time > cooldown):

                    print(f"Accepted prediction -> raw: '{raw_pred}'  cleaned: '{clean_pred}'")

                    if clean_pred in special_gestures:
                        phrase = special_gestures[clean_pred]
                        sentence += phrase + " "
                        engine.say(phrase)
                        engine.runAndWait()
                        current_word = ""  # reset letters
                    else:
                        current_word += clean_pred

                    last_prediction = clean_pred
                    last_letter_time = time.time()
                    latest_prediction = clean_pred

            # Draw bounding box
            try:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, clean_pred if clean_pred != "" else normalize_raw(raw_pred),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception:
                pass

        # Finalize word after 5s pause
        if current_word != "" and (time.time() - last_letter_time > 5):
            sentence += current_word + " "
            engine.say(current_word)
            engine.runAndWait()
            current_word = ""
            last_prediction = None

        # Show ongoing sentence
        display_text = sentence + current_word
        #cv2.putText(frame, display_text, (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app = Flask(__name__)
CORS(app)

from flask import jsonify

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global latest_prediction
    return jsonify({"prediction": latest_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
