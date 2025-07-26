import cv2
import mediapipe as mp
import time
import google.generativeai as genai
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import argparse

import torch
from torch.serialization import safe_globals
from ultralytics.nn.tasks import DetectionModel

# gemini api key setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# Ball detection enabled with new implementation
ball_detection_enabled = True
print("✅ Ball detection enabled (Red-based)")

import cv2
import numpy as np

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None


def calculate_ball_wrist_metrics(ball_pos, wrist_pos, frame_shape):
    """Calculate relationship between ball and wrist positions"""
    if ball_pos is None or wrist_pos is None:
        return None
    
    # Convert wrist position from normalized to pixel coordinates
    h, w = frame_shape[:2]
    wrist_x, wrist_y = int(wrist_pos[0] * w), int(wrist_pos[1] * h)
    ball_x, ball_y = ball_pos
    
    # Calculate distance
    import math
    distance = math.sqrt((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)
    
    # Calculate angle (in degrees, 0 = directly right, 90 = directly up)
    angle = math.degrees(math.atan2(wrist_y - ball_y, ball_x - wrist_x))
    
    # Relative position description
    dx, dy = ball_x - wrist_x, ball_y - wrist_y
    if abs(dx) > abs(dy):
        horizontal = "right" if dx > 0 else "left"
        vertical = "above" if dy < 0 else "below"
        position = f"{horizontal}-{vertical}"
    else:
        vertical = "above" if dy < 0 else "below"
        horizontal = "right" if dx > 0 else "left"  
        position = f"{vertical}-{horizontal}"
    
    return {
        "distance": distance,
        "angle": angle,
        "position": position,
        "ball_coords": ball_pos,
        "wrist_coords": (wrist_x, wrist_y)
    }

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

def generate_comparison(pose_sequence1, pose_sequence2):
    pose_text1 = ""
    for i, pose in enumerate(pose_sequence1):
        pose_text1 += f"Frame {i+1}: " + ", ".join([
            f"{name}({round(x,2)}, {round(y,2)})" for name, (x, y) in pose.items()
        ]) + "\n"
        
    pose_text2 = ""
    for i, pose in enumerate(pose_sequence2):
        pose_text2 += f"Frame {i+1}: " + ", ".join([
            f"{name}({round(x,2)}, {round(y,2)})" for name, (x, y) in pose.items()
        ]) + "\n"

    prompt = f"""
You are a basketball coach helping a player improve their shooting form. You have two shooting sequences:

PLAYER'S CURRENT FORM (Video 1):
{pose_text1}

TARGET BENCHMARK FORM (Video 2 - Professional/Reference):
{pose_text2}

The player is trying to emulate the benchmark form (Video 2). Please evaluate the player's current form (Video 1) against this target and provide coaching feedback.

Please provide:
1. ACCURACY SCORE: Rate how well the player replicates the benchmark form (0-100, where 100 = perfect replication)
2. PHASE-BY-PHASE ANALYSIS: Break down key differences in:
   - Starting position and setup
   - Upward motion and coordination  
   - Release point and peak form
   - Follow-through technique
3. SPECIFIC CORRECTIONS: What exactly should the player change to better match the benchmark?
4. PRIORITY IMPROVEMENTS: Which 2-3 changes would have the biggest impact?

Format your response as:
ACCURACY SCORE: [0-100]
ANALYSIS: [detailed phase-by-phase breakdown comparing player to benchmark]
CORRECTIONS: [specific actionable changes needed]
PRIORITIES: [top 2-3 most important improvements]
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

class SimpleShootingDetector:
    def __init__(self, buffer_size=1000):
        self.pose_buffer = []
        self.buffer_size = buffer_size
        self.state = "waiting"  # waiting, start_detected, tracking, peak_reached, completed, finished
        self.start_frame_index = 0
        self.peak_wrist_y = None
        self.frames_since_peak = 0
        self.shooting_completed = False
        self.video_ended = False
        
        # Track key transition points for alignment
        self.transition_points = {
            "start_detected": -1,
            "tracking": -1, 
            "peak_reached": -1,
            "completed": -1
        }
        self.current_frame_count = 0
        
    def add_pose(self, pose):
        # Add new pose to buffer
        self.pose_buffer.append(pose)
        
        # Maintain buffer size
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
            # Adjust start_frame_index if we're tracking
            if self.state != "waiting" and self.start_frame_index > 0:
                self.start_frame_index -= 1
            elif self.state != "waiting" and self.start_frame_index == 0:
                # Start frame was removed, reset detection
                self.reset_detection()
        
        # Run state machine
        result = self.process_state()
        self.current_frame_count += 1
        return result
    
    def add_pose_with_ball(self, pose, ball_pos=None):
        """Add pose with optional ball position for enhanced detection"""
        # Add new pose to buffer
        self.pose_buffer.append(pose)
        
        # Maintain buffer size
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
            # Adjust start_frame_index if we're tracking
            if self.state != "waiting" and self.start_frame_index > 0:
                self.start_frame_index -= 1
            elif self.state != "waiting" and self.start_frame_index == 0:
                # Start frame was removed, reset detection
                self.reset_detection()
        
        # Run state machine with ball position
        result = self.process_state(ball_pos)
        self.current_frame_count += 1
        return result
    
    def finalize_on_video_end(self):
        """Call this when video ends to finalize any incomplete shooting sequence"""
        if self.state in ["tracking", "peak_reached"] and not self.shooting_completed:
            # Video ended while tracking - capture what we have
            shooting_sequence = self.pose_buffer[self.start_frame_index:]
            print(f"🏀 Video ended → COMPLETED! (forced end, {len(shooting_sequence)} frames captured)")
            self.shooting_completed = True
            self.state = "finished"
            return True, shooting_sequence
        return False, None
        
    def process_state(self, ball_pos=None):
        if len(self.pose_buffer) < 3:
            return False, "waiting"
            
        # If shooting already completed, just ignore subsequent frames
        if self.shooting_completed:
            return False, "finished"
            
        current_pose = self.pose_buffer[-1]
        wrist_y = current_pose["right_wrist"][1]
        elbow_y = current_pose["right_elbow"][1] 
        shoulder_y = current_pose["right_shoulder"][1]
        
        if self.state == "waiting":
            return self.check_start_position(wrist_y, elbow_y, shoulder_y)
        elif self.state == "start_detected":
            return self.check_upward_movement(wrist_y, elbow_y)
        elif self.state == "tracking":
            return self.track_motion(wrist_y, elbow_y, shoulder_y, ball_pos)
        elif self.state == "peak_reached":
            return self.check_completion(wrist_y, elbow_y, shoulder_y)
            
        return False, self.state
        
    def check_start_position(self, wrist_y, elbow_y, shoulder_y):
        # Check if we're in starting position: wrist >= elbow AND elbow > shoulder
        in_starting_position = (wrist_y >= elbow_y - 0.05) and (elbow_y > shoulder_y)
        
        if in_starting_position:
            print("🏀 State: waiting → start_detected (found starting position)")
            self.state = "start_detected"
            self.start_frame_index = len(self.pose_buffer) - 1
            self.transition_points["start_detected"] = self.current_frame_count
            return False, "start_detected"
        return False, "waiting"
        
    def find_different_frame(self, current_wrist_y, current_elbow_y, threshold=0.001):
        """Find the most recent frame that's different from current pose"""
        for i in range(len(self.pose_buffer) - 2, -1, -1):  # Go backwards from second-to-last
            frame_wrist_y = self.pose_buffer[i]["right_wrist"][1]
            frame_elbow_y = self.pose_buffer[i]["right_elbow"][1]
            
            wrist_diff = abs(current_wrist_y - frame_wrist_y)
            elbow_diff = abs(current_elbow_y - frame_elbow_y)
            
            if wrist_diff > threshold or elbow_diff > threshold:
                return self.pose_buffer[i], i
                
        # If no different frame found, return the oldest frame
        return self.pose_buffer[0], 0
    
    def check_upward_movement(self, wrist_y, elbow_y):
        if len(self.pose_buffer) >= self.start_frame_index + 3:
            # Find the most recent frame that's different from current
            different_pose, frame_index = self.find_different_frame(wrist_y, elbow_y)
            prev_wrist_y = different_pose["right_wrist"][1]
            prev_elbow_y = different_pose["right_elbow"][1]
            
            wrist_moved_up = wrist_y < prev_wrist_y
            elbow_moved_up = elbow_y < prev_elbow_y

            print(f"Comparing to frame {frame_index}: prev_wrist_y {prev_wrist_y:.3f} → wrist_y {wrist_y:.3f}")
            print(f"wrist_moved_up? {wrist_moved_up}, elbow_moved_up? {elbow_moved_up}")
            
            if wrist_moved_up and elbow_moved_up:
                print("🏀 State: start_detected → tracking (upward movement confirmed)")
                self.state = "tracking"
                self.peak_wrist_y = wrist_y
                self.transition_points["tracking"] = self.current_frame_count
                return False, "tracking"
        return False, "start_detected"
        
    def track_motion(self, wrist_y, elbow_y, shoulder_y, ball_pos=None):
        # Update peak wrist position
        if wrist_y < self.peak_wrist_y:
            self.peak_wrist_y = wrist_y
            self.frames_since_peak = 0
        else:
            self.frames_since_peak += 1
        
        # If ball is detected, check if ball has left the wrist area (release point)
        if ball_pos is not None:
            current_pose = self.pose_buffer[-1] if self.pose_buffer else None
            if current_pose and "right_wrist" in current_pose:
                wrist_pos = current_pose["right_wrist"]
                # Calculate ball-wrist distance in normalized coordinates
                ball_norm_x = ball_pos[0] / 640  # Assuming resized frame width
                ball_norm_y = ball_pos[1] / 480  # Assuming resized frame height
                wrist_x, wrist_y_pos = wrist_pos[0], wrist_pos[1]
                
                distance = ((ball_norm_x - wrist_x)**2 + (ball_norm_y - wrist_y_pos)**2)**0.5
                
                # If ball is far enough from wrist AND wrist is above shoulder, consider it released (peak reached)
                if distance > 0.15 and wrist_y < shoulder_y:  # Ball release + wrist above shoulder
                    print(f"🏀 State: tracking → peak_reached (ball released, distance: {distance:.3f}, wrist above shoulder)")
                    self.state = "peak_reached"
                    self.transition_points["peak_reached"] = self.current_frame_count
                    return False, "peak_reached"
        
        # Fallback to original logic if no ball detected or ball still close to wrist
        # Check if wrist has reached peak (stopped going up for a few frames) AND wrist is above shoulder
        if self.frames_since_peak >= 3 and wrist_y < shoulder_y:
            print(f"🏀 State: tracking → peak_reached (wrist peak at {self.peak_wrist_y:.3f}, wrist above shoulder)")
            self.state = "peak_reached"
            self.transition_points["peak_reached"] = self.current_frame_count
            return False, "peak_reached"
            
        return False, "tracking"
        
    def check_completion(self, wrist_y, elbow_y, shoulder_y):
        # Check if elbow has dropped below shoulder (shooting completion)
        if wrist_y > shoulder_y:
            # Shooting motion complete! Return sequence from start
            shooting_sequence = self.pose_buffer[self.start_frame_index:]
            print(f"🏀 State: peak_reached → COMPLETED! (wrist below shoulder, {len(shooting_sequence)} frames captured)")
            self.shooting_completed = True  # Mark as completed to ignore subsequent frames
            self.state = "finished"
            self.transition_points["completed"] = self.current_frame_count
            return True, shooting_sequence
            
        # Timeout if we've been waiting too long for completion
        if self.frames_since_peak > 300:  # Increased timeout for follow-through
            print("🏀 State: peak_reached → timeout (elbow didn't drop)")
            self.shooting_completed = True  # Also mark as completed on timeout
            self.state = "finished"
            return False, "timeout"
            
        self.frames_since_peak += 1
        return False, "peak_reached"
        
    def is_straight_line_alignment(self, wrist_y, elbow_y, shoulder_y):
        # Check if joints are roughly in a straight line (vertically aligned)
        # Allow some tolerance for natural human movement
        wrist_elbow_diff = abs(wrist_y - elbow_y)
        elbow_shoulder_diff = abs(elbow_y - shoulder_y)
        
        # They should be roughly aligned vertically (small y differences)
        return wrist_elbow_diff < 0.15 and elbow_shoulder_diff < 0.15 and wrist_y < shoulder_y
        
    def reset_detection(self):
        print("🏀 State: → waiting (reset)")
        self.state = "waiting"
        self.start_frame_index = 0
        self.peak_wrist_y = None
        self.frames_since_peak = 0
        self.shooting_completed = False
        self.video_ended = False
        self.transition_points = {
            "start_detected": -1,
            "tracking": -1, 
            "peak_reached": -1,
            "completed": -1
        }
        self.current_frame_count = 0
        
def extract_shooting_sequence_with_display(video_path, window_name, x_offset=0):
    """Extract shooting sequence from a video file with visual display"""
    print(f"Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None, None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None, None
    
    motion_detector = SimpleShootingDetector()
    extracted_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video ended - check if we need to finalize incomplete shooting
            shooting_detected, result = motion_detector.finalize_on_video_end()
            if shooting_detected and isinstance(result, list):
                print(f"Video ended - extracted shooting sequence with {len(result)} frames")
                print(f"Transition points: {motion_detector.transition_points}")
                cap.release()
                cv2.destroyWindow(window_name)
                return result, extracted_frames, motion_detector.transition_points
            break
            
        # Resize frame for side-by-side display
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, _ = frame.shape
        ball_position = detect_ball(frame)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            
            # Ball detection for release point analysis (no visual marking)
            if ball_position:
                norm_ball = (ball_position[0] / w, ball_position[1] / h)
                keypoints["ball"] = norm_ball
            
            # Add pose to detector and check for shooting (with ball position if available)
            shooting_detected, result = motion_detector.add_pose_with_ball(keypoints, ball_position)
            
            # Display current detection state
            if isinstance(result, str):
                cv2.putText(frame, f"Status: {result}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Status: ready", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw keypoints with transparency
            h, w, _ = frame.shape
            overlay = frame.copy()
            for name, (x, y) in keypoints.items():
                cx, cy = int(x * w), int(y * h)
                if "wrist" in name:
                    color = (0, 255, 255)
                elif "ankle" in name:
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(overlay, (cx, cy), 3, color, -1)
            # Blend overlay with original frame (0.4 opacity for keypoints)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Store frame if we're in shooting sequence
            if motion_detector.state in ["start_detected", "tracking", "peak_reached"]:
                extracted_frames.append(frame.copy())
            
            if shooting_detected and isinstance(result, list):
                print(f"Shooting sequence extracted with {len(result)} frames")
                print(f"Transition points: {motion_detector.transition_points}")
                cap.release()
                cv2.destroyWindow(window_name)
                return result, extracted_frames, motion_detector.transition_points
        
        # Add video title
        cv2.putText(frame, window_name, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        cv2.moveWindow(window_name, x_offset, 100)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    print("No shooting sequence detected in video")
    return None, None, None

def extract_shooting_sequence(video_path):
    """Extract shooting sequence from a video file (no display)"""
    print(f"Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None
    
    motion_detector = SimpleShootingDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video ended - check if we need to finalize incomplete shooting
            shooting_detected, result = motion_detector.finalize_on_video_end()
            if shooting_detected and isinstance(result, list):
                print(f"Video ended - extracted shooting sequence with {len(result)} frames")
                cap.release()
                return result
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            shooting_detected, result = motion_detector.add_pose(keypoints)
            
            if shooting_detected and isinstance(result, list):
                print(f"Shooting sequence extracted with {len(result)} frames")
                cap.release()
                return result
    
    cap.release()
    print("No shooting sequence detected in video")
    return None

def align_sequences_by_transitions(frames1, frames2, transitions1, transitions2):
    """Align two sequences based on their key transition points"""
    print(f"\n4. Aligning sequences based on transition points...")
    print(f"Video 1 transitions: {transitions1}")
    print(f"Video 2 transitions: {transitions2}")
    
    # Define key phases and their relative positions
    phases = ["start_detected", "tracking", "peak_reached", "completed"]
    
    # Extract valid transitions (ignore -1 values)
    valid_transitions1 = {k: v for k, v in transitions1.items() if v >= 0}
    valid_transitions2 = {k: v for k, v in transitions2.items() if v >= 0}
    
    if not valid_transitions1 or not valid_transitions2:
        print("Warning: Not enough transition points for alignment, using simple alignment")
        return frames1, frames2
    
    # Calculate segments between transitions
    aligned_frames1 = []
    aligned_frames2 = []
    
    # Get phase boundaries for both sequences
    phases1 = [(phase, transitions1.get(phase, -1)) for phase in phases if transitions1.get(phase, -1) >= 0]
    phases2 = [(phase, transitions2.get(phase, -1)) for phase in phases if transitions2.get(phase, -1) >= 0]
    
    if not phases1 or not phases2:
        return frames1, frames2
    
    # Align each phase segment
    prev_frame1, prev_frame2 = 0, 0
    
    for i, (phase, _) in enumerate(phases1):
        if phase not in [p[0] for p in phases2]:
            continue
            
        # Find frame indices for this phase in both sequences
        frame1_idx = transitions1[phase] if phase in transitions1 and transitions1[phase] >= 0 else len(frames1) - 1
        frame2_idx = transitions2[phase] if phase in transitions2 and transitions2[phase] >= 0 else len(frames2) - 1
        
        # Calculate segment lengths
        segment1_length = frame1_idx - prev_frame1
        segment2_length = frame2_idx - prev_frame2
        
        # Get frames for this segment
        segment1 = frames1[prev_frame1:frame1_idx] if prev_frame1 < len(frames1) else []
        segment2 = frames2[prev_frame2:frame2_idx] if prev_frame2 < len(frames2) else []
        
        # Align segments by stretching the shorter one
        target_length = max(segment1_length, segment2_length, 1)
        
        aligned_segment1 = stretch_segment(segment1, target_length)
        aligned_segment2 = stretch_segment(segment2, target_length)
        
        aligned_frames1.extend(aligned_segment1)
        aligned_frames2.extend(aligned_segment2)
        
        prev_frame1 = frame1_idx
        prev_frame2 = frame2_idx
    
    print(f"Aligned sequences: {len(aligned_frames1)} frames each")
    return aligned_frames1, aligned_frames2

def stretch_segment(frames, target_length):
    """Stretch a segment to target length by repeating frames"""
    if not frames or target_length <= 0:
        return frames
    
    if len(frames) >= target_length:
        return frames[:target_length]
    
    stretched = []
    for i in range(target_length):
        # Map target index to original frame index
        orig_idx = min(int(i * len(frames) / target_length), len(frames) - 1)
        stretched.append(frames[orig_idx])
    
    return stretched

def calculate_optimal_crop_region(frames):
    """Calculate the optimal crop region for the entire video sequence"""
    all_min_x, all_max_x = [], []
    all_min_y, all_max_y = [], []
    
    # Analyze pose landmarks across all frames
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Extract coordinates of visible landmarks
            x_coords = [int(lm.x * w) for lm in landmarks if lm.visibility > 0.5]
            y_coords = [int(lm.y * h) for lm in landmarks if lm.visibility > 0.5]
            
            if x_coords and y_coords:
                all_min_x.append(min(x_coords))
                all_max_x.append(max(x_coords))
                all_min_y.append(min(y_coords))
                all_max_y.append(max(y_coords))
    
    if not all_min_x:  # No pose detected in any frame
        return None
    
    # Find overall bounding box across entire sequence
    overall_min_x = min(all_min_x)
    overall_max_x = max(all_max_x)
    overall_min_y = min(all_min_y)
    overall_max_y = max(all_max_y)
    
    # Add padding
    width = overall_max_x - overall_min_x
    height = overall_max_y - overall_min_y
    padding_x = int(width * 0.3)
    padding_y = int(height * 0.2)
    
    h, w = frames[0].shape[:2]
    crop_x1 = max(0, overall_min_x - padding_x)
    crop_y1 = max(0, overall_min_y - padding_y)
    crop_x2 = min(w, overall_max_x + padding_x)
    crop_y2 = min(h, overall_max_y + padding_y)
    
    # Ensure minimum aspect ratio of 0.67 (width/height >= 0.67)
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    current_ratio = crop_width / crop_height
    
    if current_ratio < 0.67:
        # Need to increase width to meet minimum ratio
        target_width = int(crop_height * 0.67)
        width_increase = target_width - crop_width
        
        # Expand width symmetrically if possible
        left_expand = width_increase // 2
        right_expand = width_increase - left_expand
        
        new_x1 = max(0, crop_x1 - left_expand)
        new_x2 = min(w, crop_x2 + right_expand)
        
        # If we hit boundaries, adjust the other side
        if new_x1 == 0:
            new_x2 = min(w, new_x2 + (crop_x1 - new_x1))
        if new_x2 == w:
            new_x1 = max(0, new_x1 - (new_x2 - crop_x2))
            
        crop_x1, crop_x2 = new_x1, new_x2
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def apply_consistent_crop(frame, crop_region):
    """Apply consistent crop region to a frame"""
    if crop_region is None:
        return frame
    
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    
    # Simply crop the frame - no resizing to maintain natural dimensions
    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped

def play_sequences_side_by_side(frames1, frames2, video1_name, video2_name, transitions1=None, transitions2=None):
    """Display both shooting video sequences side by side for visual comparison"""
    print(f"\n5. Playing aligned shooting sequences side by side...")
    print("Press 'q' to skip, any other key to advance frame by frame")
    
    if not frames1 or not frames2:
        print("Error: No video frames available for comparison")
        return
    
    # Align sequences if transition data is available
    if transitions1 and transitions2:
        frames1, frames2 = align_sequences_by_transitions(frames1, frames2, transitions1, transitions2)
    
    # Calculate optimal crop regions for consistent framing throughout each video
    print("Calculating optimal crop regions for consistent framing...")
    crop_region1 = calculate_optimal_crop_region(frames1)
    crop_region2 = calculate_optimal_crop_region(frames2)
    
    max_frames = max(len(frames1), len(frames2))
    
    for i in range(max_frames):
        # Get frames (repeat last frame if one sequence is shorter)
        frame1 = frames1[min(i, len(frames1)-1)].copy()
        frame2 = frames2[min(i, len(frames2)-1)].copy()
        
        # Apply consistent crop regions (each maintains its own natural width)
        frame1 = apply_consistent_crop(frame1, crop_region1)
        frame2 = apply_consistent_crop(frame2, crop_region2)
        
        # Add titles to frames
        cv2.putText(frame1, video1_name, (10, frame1.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame2, video2_name, (10, frame2.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Ensure both frames have the same height for side-by-side display
        h1, h2 = frame1.shape[0], frame2.shape[0]
        if h1 != h2:
            target_height = max(h1, h2)
            if h1 < target_height:
                frame1 = cv2.resize(frame1, (frame1.shape[1], target_height))
            if h2 < target_height:
                frame2 = cv2.resize(frame2, (frame2.shape[1], target_height))
        
        # Combine frames side by side
        combined_frame = np.hstack((frame1, frame2))
        
        # Add frame counter
        cv2.putText(combined_frame, f"Frame: {i+1}/{max_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Shooting Comparison", combined_frame)
        cv2.moveWindow("Shooting Comparison", 50, 50)
        
        key = cv2.waitKey(100)  # Auto-advance after 167ms (3x faster)
        if key & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Shooting Comparison")

def create_skeleton_frame(pose_data, title, size):
    """Create a frame showing skeleton visualization from pose data"""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    h, w = size[1], size[0]
    
    # Draw title
    cv2.putText(frame, title, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    
    # Draw skeleton connections
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), 
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle")
    ]
    
    # Draw connections
    for joint1, joint2 in connections:
        if joint1 in pose_data and joint2 in pose_data:
            x1, y1 = int(pose_data[joint1][0] * w), int(pose_data[joint1][1] * h)
            x2, y2 = int(pose_data[joint2][0] * w), int(pose_data[joint2][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Draw keypoints
    for name, (x, y) in pose_data.items():
        if name == "ball":
            continue
        cx, cy = int(x * w), int(y * h)
        if "wrist" in name:
            color = (0, 255, 255)  # Yellow for wrists
        elif "ankle" in name:
            color = (255, 0, 255)  # Magenta for ankles  
        elif "shoulder" in name:
            color = (0, 255, 0)    # Green for shoulders
        elif "elbow" in name:
            color = (255, 0, 0)    # Blue for elbows
        else:
            color = (128, 128, 128) # Gray for others
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 4, color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    return frame

def compare_videos(video1_path, video2_path):
    """Compare shooting forms from two videos with visual display"""
    print("=== BASKETBALL SHOOTING COMPARISON ===")
    print("Press 'q' to quit during video processing")
    
    # Extract shooting sequences from both videos with visual display
    print("\n1. Extracting shooting sequence from Video 1...")
    sequence1, frames1, transitions1 = extract_shooting_sequence_with_display(video1_path, "Video 1", x_offset=0)
    
    print("\n2. Extracting shooting sequence from Video 2...")
    sequence2, frames2, transitions2 = extract_shooting_sequence_with_display(video2_path, "Video 2", x_offset=660)
    
    if sequence1 is None:
        print("Error: No shooting sequence found in Video 1")
        return
        
    if sequence2 is None:
        print("Error: No shooting sequence found in Video 2")
        return
    
    print(f"\n3. Sequences extracted (Video 1: {len(sequence1)} frames, Video 2: {len(sequence2)} frames)")
    
    # Show actual video sequences side by side with keypoints (aligned by transitions)
    play_sequences_side_by_side(frames1, frames2, "Video 1 Sequence", "Video 2 Sequence", transitions1, transitions2)
    
    print(f"\n6. Generating AI coaching feedback...")
    print("Analyzing player's form against benchmark...")
    
    # Generate comparison using Gemini (player vs benchmark)
    coaching_feedback = generate_comparison(sequence1, sequence2)
    
    print("\n=== COACHING FEEDBACK ===")
    print("📊 Player's Form vs Professional Benchmark")
    print("="*50)
    print(coaching_feedback)

def is_shooting_motion(pose_seq):
    # Legacy function - kept for compatibility but not used
    if len(pose_seq) < 10:
        return False
    wrist_y = [pose["right_wrist"][1] for pose in pose_seq]
    shoulder_y = [pose["right_shoulder"][1] for pose in pose_seq]
    wrist_above_shoulder = [w < s for w, s in zip(wrist_y, shoulder_y)]
    return sum(wrist_above_shoulder) > 5

def main():
    parser = argparse.ArgumentParser(description='Basketball Feedback Generator')
    parser.add_argument('--video', '-v', type=str, help='Path to video file (default: use webcam)')
    parser.add_argument('--compare', '-c', type=str, help='Path to second video for comparison')
    args = parser.parse_args()
    
    # If comparison mode, process both videos
    if args.compare:
        if not args.video:
            print("Error: --video is required when using --compare")
            return
        compare_videos(args.video, args.compare)
        return
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file '{args.video}' not found.")
            return
        cap = cv2.VideoCapture(args.video)
        print(f"Processing video: {args.video}. Press 'q' to quit.")
    else:
        cap = cv2.VideoCapture(0)
        print("Starting webcam. Press 'q' to quit.")
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    feedback_displayed = False
    motion_detector = SimpleShootingDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video ended - check if we need to finalize incomplete shooting
            if not feedback_displayed:
                shooting_detected, result = motion_detector.finalize_on_video_end()
                if shooting_detected and isinstance(result, list):
                    print(f"Video ended - finalizing incomplete shooting motion ({len(result)} frames). Sending to Gemini.")
                    feedback = generate_feedback(result)
                    print("Gemini Feedback:\n", feedback)
                    feedback_displayed = True
            break

        if not args.video:
            frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        h, w, _ = frame.shape
        ball_position = detect_ball(frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            keypoints = extract_keypoints(results.pose_landmarks.landmark)

            # Ball detection for release point analysis (no visual marking)
            if ball_position:
                norm_ball = (ball_position[0] / w, ball_position[1] / h)
                keypoints["ball"] = norm_ball

            # Add pose to detector and check for shooting (with ball position if available)
            shooting_detected, result = motion_detector.add_pose_with_ball(keypoints, ball_position)
            
            if shooting_detected and isinstance(result, list) and not feedback_displayed:
                print(f"Shooting motion detected! Captured {len(result)} frames. Sending to Gemini.")
                feedback = generate_feedback(result)
                print("Gemini Feedback:\n", feedback)
                feedback_displayed = True
                feedback_time = time.time()

            # Display current detection state
            if isinstance(result, str):
                cv2.putText(frame, f"Status: {result}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Status: ready", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            overlay = frame.copy()
            for name, (x, y) in keypoints.items():
                cx, cy = int(x * w), int(y * h)
                if name == "ball":
                    continue
                elif "wrist" in name:
                    color = (0, 255, 255)
                elif "ankle" in name:
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(overlay, (cx, cy), 3, color, -1)
            # Blend overlay with original frame (0.4 opacity for keypoints)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

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
            # Don't clear pose_buffer - let motion continue

        cv2.putText(frame, "Live Shot Analyzer", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Live Shot Analyzer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
