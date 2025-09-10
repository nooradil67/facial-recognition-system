import cv2
import numpy as np
import os
import csv
from datetime import datetime
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import threading
import time

app = Flask(__name__)

# Create necessary directories
if not os.path.exists("faces"):
    os.makedirs("faces")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables for camera handling
camera = None
camera_lock = threading.Lock()
current_mode = None
registration_name = ""
saved_count = 0
required_samples = 10
attendance_name = ""
attendance_marked = False

def get_face_landmarks(gray_face):
    """Simple function to get facial landmarks (eyes) for alignment"""
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
    
    if len(eyes) >= 2:
        # Sort eyes by x-coordinate
        eyes = sorted(eyes, key=lambda x: x[0])
        return eyes
    return None

def align_face(face_img):
    """Align face based on eye positions"""
    gray = face_img if len(face_img.shape) == 2 else cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    landmarks = get_face_landmarks(gray)
    
    if landmarks is None or len(landmarks) < 2:
        return face_img  # Return original if can't detect eyes
    
    # Get eye centers
    left_eye = (landmarks[0][0] + landmarks[0][2]//2, landmarks[0][1] + landmarks[0][3]//2)
    right_eye = (landmarks[1][0] + landmarks[1][2]//2, landmarks[1][1] + landmarks[1][3]//2)
    
    # Calculate angle between the eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Calculate center point between eyes (convert to float)
    eyes_center = (float((left_eye[0] + right_eye[0]) // 2), float((left_eye[1] + right_eye[1]) // 2))
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    
    # Apply rotation
    aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), 
                                 flags=cv2.INTER_CUBIC)
    
    return aligned_face

def compare_faces(face1, face2):
    """Compare two faces using multiple methods for better accuracy"""
    # Resize faces to same dimensions
    face1 = cv2.resize(face1, (200, 200))
    face2 = cv2.resize(face2, (200, 200))
    
    # Method 1: Histogram comparison
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)
    
    # Compare using correlation
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Method 2: Template matching
    result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
    template_similarity = np.max(result)

    
    # Combined similarity score (weighted average)
    similarity = 0.7 * hist_similarity + 0.3 * template_similarity
    
    return similarity

def init_camera():
    """Initialize the camera"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Allow camera to warm up
        time.sleep(2)

def release_camera():
    """Release the camera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames(mode):
    """Generate frames for video feed based on mode"""
    global saved_count, registration_name, attendance_marked, attendance_name
    
    init_camera()
    count = 0
    no_match_count = 0
    max_attempts = 50
    
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if mode == 'register':
            # Display instructions on frame
            cv2.putText(frame, "Look straight at the camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {saved_count}/{required_samples}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(frame, "Multiple faces detected", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                (x, y, w, h) = faces[0]
                
                # Check if face is reasonably centered and sized
                height, width = frame.shape[:2]
                center_x, center_y = width//2, height//2
                face_center_x, face_center_y = x + w//2, y + h//2
                
                # Calculate distance from center
                dist_from_center = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
                max_allowed_dist = min(width, height) * 0.2
                
                if dist_from_center > max_allowed_dist:
                    cv2.putText(frame, "Please center your face", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif w < 100 or h < 100:
                    cv2.putText(frame, "Move closer to the camera", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Good position! Hold still", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save face every 10 frames
                    count += 1
                    if count % 10 == 0 and saved_count < required_samples:
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Align the face before saving
                        try:
                            aligned_face = align_face(face_roi)
                        except:
                            aligned_face = face_roi  # Use original if alignment fails
                        
                        # Resize to standard size for consistency
                        aligned_face = cv2.resize(aligned_face, (200, 200))
                        
                        cv2.imwrite(f"faces/{registration_name}_{saved_count}.jpg", aligned_face)
                        saved_count += 1
                        print(f"Saved sample {saved_count}/{required_samples}")
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        elif mode == 'attendance':
            # Display attempt count
            cv2.putText(frame, f"Attempts: {no_match_count}/{max_attempts}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(faces) == 0:
                cv2.putText(frame, "No face detected - Please position face in frame", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                no_match_count += 1
            elif len(faces) > 1:
                cv2.putText(frame, "Multiple faces detected - Ensure only one person in frame", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                no_match_count += 1
            else:
                (x, y, w, h) = faces[0]
                
                # Check if face is reasonably centered and sized (similar to registration)
                height, width = frame.shape[:2]
                center_x, center_y = width//2, height//2
                face_center_x, face_center_y = x + w//2, y + h//2
                
                # Calculate distance from center
                dist_from_center = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
                max_allowed_dist = min(width, height) * 0.2
                
                if dist_from_center > max_allowed_dist:
                    cv2.putText(frame, "Please center your face in the frame", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    no_match_count += 1
                elif w < 100 or h < 100:
                    cv2.putText(frame, "Move closer to the camera", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    no_match_count += 1
                else:
                    cv2.putText(frame, "Good position! Analyzing face...", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Extract and align the face
                    current_face = gray[y:y+h, x:x+w]
                    try:
                        aligned_face = align_face(current_face)
                    except:
                        aligned_face = current_face  # Use original if alignment fails
                    aligned_face = cv2.resize(aligned_face, (200, 200))
                    
                    # Load registered faces
                    known_faces = []
                    known_names = []
                    
                    for filename in os.listdir("faces"):
                        if filename.endswith(".jpg"):
                            img = cv2.imread(f"faces/{filename}", cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                known_faces.append(img)
                                known_names.append(filename.split("_")[0])
                    
                    best_match = None
                    best_similarity = -1
                    
                    # Compare with all known faces
                    for i, known_face in enumerate(known_faces):
                        similarity = compare_faces(aligned_face, known_face)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = known_names[i]
                    
                    # Display similarity score for debugging
                    cv2.putText(frame, f"Best match: {best_match} ({best_similarity:.2f})", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Check if we have a good match
                    if best_similarity > 0.65:  # Slightly increased threshold
                        # Save to CSV
                        with open('attendance.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([best_match, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                        
                        cv2.putText(frame, f"Attendance marked for {best_match}", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"Attendance marked for {best_match}")
                        attendance_marked = True
                        attendance_name = best_match
                    else:
                        cv2.putText(frame, "No match found - Try again or register new face", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        no_match_count += 1
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if no_match_count >= max_attempts or attendance_marked:
                time.sleep(2)  # Give time to see the result
                break
        
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global registration_name, saved_count
    
    if request.method == 'POST':
        registration_name = request.form['name']
        saved_count = 0
        return render_template('register_camera.html', name=registration_name)
    
    return render_template('register.html')

@app.route('/video_feed_register')
def video_feed_register():
    global current_mode
    current_mode = 'register'
    return Response(generate_frames('register'), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registration_status')
def registration_status():
    global saved_count
    if saved_count >= required_samples:
        return jsonify({'status': 'complete', 'count': saved_count})
    return jsonify({'status': 'in_progress', 'count': saved_count})

@app.route('/registration_success')
def registration_success():
    global saved_count, registration_name
    count = saved_count
    saved_count = 0
    return render_template('registration_success.html', name=registration_name, count=count)

@app.route('/attendance')
def attendance():
    global attendance_marked
    attendance_marked = False
    return render_template('camera.html')

@app.route('/video_feed_attendance')
def video_feed_attendance():
    global current_mode
    current_mode = 'attendance'
    return Response(generate_frames('attendance'), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_status')
def attendance_status():
    global attendance_marked, attendance_name
    if attendance_marked:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({'status': 'marked', 'name': attendance_name, 'timestamp': timestamp})
    return jsonify({'status': 'in_progress'})

@app.route('/attendance_success')
def attendance_success():
    global attendance_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('attendance_success.html', name=attendance_name, timestamp=timestamp)

@app.route('/view_attendance')
def view_attendance():
    attendance_data = []
    if os.path.exists('attendance.csv'):
        with open('attendance.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    attendance_data.append(row)
    
    return render_template('attendance.html', attendance_data=attendance_data)

@app.route('/cleanup')
def cleanup():
    """Clean up camera resources"""
    release_camera()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create attendance file if it doesn't exist
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Timestamp'])
    
    app.run(debug=True)