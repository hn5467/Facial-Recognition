import cv2
import face_recognition

# Load the reference image and compute its encoding
image_of_person = face_recognition.load_image_file("reference_image.jpg")
encodings = face_recognition.face_encodings(image_of_person)
if not encodings:
    raise SystemExit("No face found in reference_image.jpg")
known_face_encodings = [encodings[0]]

# Lower = stricter. 0.6 is the library default; 0.5 reduces false matches.
TOLERANCE = 0.5

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # BGR -> RGB (cvtColor returns a contiguous array, which dlib requires)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distance = face_recognition.face_distance(known_face_encodings, face_encoding)[0]
        is_match = distance <= TOLERANCE
        score = 1 - distance

        label = f"same person ({score:.2f})" if is_match else f"different ({score:.2f})"
        color = (0, 255, 0) if is_match else (0, 0, 255)
        print(label)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
