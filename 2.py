import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Create necessary directories
if not os.path.exists("faces"):
    os.makedirs("faces")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_new_face():
    name = input("Enter the name for the new face: ")
    print("Please look at the camera for registration...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    count = 0
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"faces/{name}_{count}.jpg", face_img)
            count += 1
        
        cv2.imshow('Registering Face - Press Q to stop', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Face registered for {name}")

def mark_attendance():
    print("Please look at the camera for attendance...")
    
    # Load registered faces
    known_faces = []
    known_names = []
    
    for filename in os.listdir("faces"):
        if filename.endswith(".jpg"):
            img = cv2.imread(f"faces/{filename}", cv2.IMREAD_GRAYSCALE)
            known_faces.append(img)
            known_names.append(filename.split("_")[0])
    
    if not known_faces:
        print("No registered faces found!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    matched = False
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            current_face = gray[y:y+h, x:x+w]
            
            # Simple face matching (you might want to improve this)
            for i, known_face in enumerate(known_faces):
                try:
                    known_face = cv2.resize(known_face, (w, h))
                    difference = cv2.absdiff(current_face, known_face)
                    similarity = np.mean(difference)
                    
                    if similarity < 50:  # Adjust this threshold as needed
                        name = known_names[i]
                        print(f"Face matched with {name}")
                        
                        # Save to CSV
                        with open('attendance.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                        
                        print("Attendance marked successfully!")
                        matched = True
                        break
                except:
                    continue
            
            if not matched:
                print("Face does not match any registered face")
        
        cv2.imshow('Marking Attendance - Press Q to stop', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or matched:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nFace Detection Attendance System")
        print("1. Register new face")
        print("2. Mark attendance")
        print("3. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            register_new_face()
        elif choice == '2':
            mark_attendance()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()