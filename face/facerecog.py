import cv2
import pathlib
import face_recognition

# Define the path to the Haar Cascade classifier XML file
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

# Load the Haar Cascade classifier
clf = cv2.CascadeClassifier(str(cascade_path))

# Load sample image and encode known face
sample_image_path = r"C:\Users\Personal Computer\face\image\reference.jpg"  # Replace with the path to your sample image
known_face_image = face_recognition.load_image_file(sample_image_path)
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

# Initialize arrays to store known faces, their names, and attendance count
known_face_encodings = [known_face_encoding]
known_face_names = ["Ian"]  # Replace with the name of the person in the sample image
attendance_count = [0]

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture each frame from the video feed
    ret, frame = camera.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found by face recognition
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Increment attendance count for the recognized face
            attendance_count[first_match_index] += 1

            # Calculate attendance percentage
            total_frames = sum(attendance_count)
            attendance_percentage = (attendance_count[first_match_index] / total_frames) * 100

            # Draw a label with the name and attendance percentage on the frame
            label = f"{name}: {attendance_percentage:.2f}%"
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
