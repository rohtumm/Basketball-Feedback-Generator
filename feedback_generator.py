import cv2
import mediapipe as mp
import time
import google.generativeai as genai
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# gemini api key setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# YOLOv8 ball detection
yolo_model = YOLO("yolov8n.pt")

def detect_ball(frame):
    results = yolo_model.predict(source=frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        if "ball" in label.lower():
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return (cx, cy) # we're only really interested in the center of the ball
    return None

# gemini feedback generation
def generate_feedback(pose_sequence):
    pose_text = ""
    for i, pose in enumerate(pose_sequence):
        pose_text += f"Frame {i+1}: " + ", ".join([
            f"{name}({round(x,2)}, {round(y,2)})" for name, (x, y) in pose.items()
        ]) + "\n"

    prompt = f"""
You are a basketball coach. Analyze this sequence of joint and ball positions from a player's shooting motion.

- Each line contains normalized (x, y) joint coordinates.
- "ball(x, y)" refers to the basketball position.
- Focus on coordination, release timing, and ball trajectory.

Sequence:
{pose_text}

Give specific, actionable feedback (max 2 sentences).
"""

    response = model.generate_content(prompt)
    return response.text

# player pose setup + extraction
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(landmarks):
    important_joints = {
        "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
        "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
        "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
        "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE
    }

    keypoints = {}
    for name, index in important_joints.items():
        landmark = landmarks[index]
        keypoints[name] = (landmark.x, landmark.y)
    return keypoints

def is_shooting_motion(pose_seq):
    if len(pose_seq) < 10:
        return False
    wrist_y = [pose["right_wrist"][1] for pose in pose_seq]
    shoulder_y = [pose["right_shoulder"][1] for pose in pose_seq]
    wrist_above_shoulder = [w < s for w, s in zip(wrist_y, shoulder_y)]
    return sum(wrist_above_shoulder) > 5

# 
cap = cv2.VideoCapture(0)
pose_buffer = []
feedback_displayed = False

print("Starting webcam. q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    h, w, _ = frame.shape
    ball_position = detect_ball(frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        keypoints = extract_keypoints(results.pose_landmarks.landmark)

        if ball_position:
            cv2.circle(frame, ball_position, 10, (0, 0, 255), 2)
            norm_ball = (ball_position[0] / w, ball_position[1] / h)
            keypoints["ball"] = norm_ball

        pose_buffer.append(keypoints)

        if len(pose_buffer) > 20:
            pose_buffer.pop(0)

        if is_shooting_motion(pose_buffer) and not feedback_displayed:
            print("Shooting motion detected. Sending to Gemini.")
            feedback = generate_feedback(pose_buffer)
            print("Gemini Feedback:\n", feedback)
            feedback_displayed = True
            feedback_time = time.time()

        for name, (x, y) in keypoints.items():
            cx, cy = int(x * w), int(y * h)
            if name == "ball":
                continue
            elif "wrist" in name: # different colors to distinguish
                color = (0, 255, 255)
            elif "ankle" in name:
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 6, color, -1)

    def draw_feedback(frame, feedback_text, x=15, y=30, max_width=620, line_height=25):
        lines = []
        for line in feedback_text.split('\n'):
            words = line.split(' ')
            current_line = ""
            for w in words:
                test_line = current_line + w + " "
                (text_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                if text_w > max_width:
                    lines.append(current_line.strip())
                    current_line = w + " "
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line.strip())

        rect_height = line_height * len(lines) + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y - line_height), (x + max_width + 10, y - line_height + rect_height), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            cv2.putText(frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


    if feedback_displayed and time.time() - feedback_time < 5:
        draw_feedback(frame, feedback)

    elif feedback_displayed and time.time() - feedback_time >= 5:
        feedback_displayed = False
        pose_buffer = []

    cv2.putText(frame, "Live Shot Analyzer", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow("Live Shot Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
