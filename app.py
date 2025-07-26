import streamlit as st
import cv2
import numpy as np
import tempfile
from feedback_generator import extract_keypoints, extract_angles_from_pose, is_shooting_motion_angles, generate_feedback, detect_ball
import mediapipe as mp
import time

st.title("Basketball Shooting Feedback Analyzer")

# Create tabs for different analysis modes
tab1, tab2 = st.tabs(["📁 Upload Video", "📹 Live Analysis"])

with tab1:
    st.write("Upload a basketball shooting video. The app will analyze the motion and provide feedback.")
    
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="upload_tab")
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        pose_buffer = []
        angle_buffer = []
        frame_count = 0
        shots_detected = []
        current_shot_frames = []
        shot_detection_cooldown = 0  # Prevent multiple detections of same shot
        
        stframe = st.empty()
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get total frames for progress
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            h, w, _ = frame.shape
            
            if results.pose_landmarks:
                keypoints = extract_keypoints(results.pose_landmarks.landmark)
                pose_buffer.append(keypoints)
                angles = extract_angles_from_pose(keypoints)
                angle_buffer.append(angles)
                if len(pose_buffer) > 20:
                    pose_buffer.pop(0)
                if len(angle_buffer) > 20:
                    angle_buffer.pop(0)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Multiple shot detection with cooldown
                if is_shooting_motion_angles(angle_buffer) and shot_detection_cooldown <= 0:
                    # Generate feedback for this shot
                    shot_feedback = generate_feedback(pose_buffer)
                    
                    # Calculate shot timing (approximate)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    shot_time = frame_count / fps
                    
                    # Store shot information
                    shot_info = {
                        'frame': frame_count,
                        'time': shot_time,
                        'feedback': shot_feedback,
                        'pose_sequence': pose_buffer.copy()
                    }
                    shots_detected.append(shot_info)
                    
                    # Add frames to current shot for visualization
                    current_shot_frames.extend(range(max(0, frame_count-30), frame_count+1))
                    
                    # Set cooldown to prevent multiple detections of same shot
                    shot_detection_cooldown = 60  # 60 frames cooldown
                    
                    # Draw shot detection indicator on frame
                    cv2.putText(frame, f"SHOT #{len(shots_detected)} DETECTED!", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {shot_time:.1f}s", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Decrease cooldown
                if shot_detection_cooldown > 0:
                    shot_detection_cooldown -= 1
                
                # Highlight current shot frames
                if frame_count in current_shot_frames:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 3)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} - Shots detected: {len(shots_detected)}")
            
            # Show frame in Streamlit
            stframe.image(frame, channels="BGR", caption=f"Frame {frame_count}")
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Display comprehensive analysis results
        if shots_detected:
            st.success(f"🎯 Analysis Complete! Detected {len(shots_detected)} shot(s)")
            
            # Display individual shot analysis
            st.subheader("Individual Shot Analysis")
            
            for i, shot in enumerate(shots_detected):
                with st.expander(f"Shot #{i+1} - {shot['time']:.1f}s"):
                    st.write(f"**Frame:** {shot['frame']}")
                    st.write(f"**Time:** {shot['time']:.1f} seconds")
                    st.write("**Feedback:**")
                    st.write(shot['feedback'])
                    
                    # Show frame range for this shot
                    start_frame = max(0, shot['frame'] - 15)
                    end_frame = shot['frame'] + 15
                    st.info(f"Shot motion detected between frames {start_frame}-{end_frame}")
        
        else:
            st.info("No shooting motions detected in this video.")
            st.write("**Tips for better detection:**")
            st.write("• Ensure clear view of the player")
            st.write("• Make sure shooting motion is visible")
            st.write("• Try different angles or lighting")

with tab2:
    st.write("Use your webcam for live basketball shooting analysis.")
    
    # Initialize session state for live analysis
    if 'live_analysis_running' not in st.session_state:
        st.session_state.live_analysis_running = False
        st.session_state.pose_buffer = []
        st.session_state.angle_buffer = []
        st.session_state.feedback_displayed = False
        st.session_state.feedback = None
        st.session_state.feedback_time = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎬 Start Live Analysis", disabled=st.session_state.live_analysis_running):
            st.session_state.live_analysis_running = True
            st.session_state.pose_buffer = []
            st.session_state.angle_buffer = []
            st.session_state.feedback_displayed = False
            st.session_state.feedback = None
    
    with col2:
        if st.button("⏹️ Stop Live Analysis", disabled=not st.session_state.live_analysis_running):
            st.session_state.live_analysis_running = False
    
    if st.session_state.live_analysis_running:
        st.info("Live analysis is running. Perform a basketball shooting motion in front of your webcam.")
        
        # Create placeholder for webcam feed
        webcam_placeholder = st.empty()
        feedback_placeholder = st.empty()
        
        # Initialize MediaPipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera permissions.")
            st.session_state.live_analysis_running = False
        else:
            try:
                while st.session_state.live_analysis_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Flip frame horizontally for mirror effect
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
                        
                        st.session_state.pose_buffer.append(keypoints)
                        angles = extract_angles_from_pose(keypoints)
                        st.session_state.angle_buffer.append(angles)
                        
                        if len(st.session_state.pose_buffer) > 20:
                            st.session_state.pose_buffer.pop(0)
                        if len(st.session_state.angle_buffer) > 20:
                            st.session_state.angle_buffer.pop(0)
                        
                        # Angle-based shooting detection
                        if is_shooting_motion_angles(st.session_state.angle_buffer) and not st.session_state.feedback_displayed:
                            st.session_state.feedback = generate_feedback(st.session_state.pose_buffer)
                            st.session_state.feedback_displayed = True
                            st.session_state.feedback_time = time.time()
                        
                        # Draw keypoints
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
                            cv2.circle(frame, (cx, cy), 6, color, -1)
                    
                    # Display feedback if available
                    if st.session_state.feedback_displayed and st.session_state.feedback:
                        if time.time() - st.session_state.feedback_time < 10:  # Show feedback for 10 seconds
                            # Draw feedback on frame
                            cv2.putText(frame, "SHOOTING MOTION DETECTED!", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            feedback_placeholder.success("🎯 Shooting motion detected!")
                            feedback_placeholder.write(st.session_state.feedback)
                        else:
                            st.session_state.feedback_displayed = False
                            st.session_state.pose_buffer = []
                            st.session_state.angle_buffer = []
                            feedback_placeholder.empty()
                    
                    # Display webcam feed
                    webcam_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")
                    
                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.1)
                    
            except Exception as e:
                st.error(f"Error during live analysis: {str(e)}")
            finally:
                cap.release()
                st.session_state.live_analysis_running = False
    else:
        st.info("Click 'Start Live Analysis' to begin real-time shooting motion detection.") 