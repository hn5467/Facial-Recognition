import cv2
from deepface import DeepFace

video_capture = cv2.VideoCapture(0)

# input files
img1_path = "pic1.jpg"
img2_path = "pic2.jpg"
img3_path = "pic3.jpg"
img4_path = "your_picture.png"

# precheck of two functionalities
print(DeepFace.verify(img1_path, img2_path))
print(DeepFace.analyze(img3_path, actions=['age', 'gender', 'race', 'emotion']))


# using your notebook camera get info about you and also check if the picture is you

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    try:
        #small_frame = cv2.resize(frame, (640, 480)) # in case you want to reduce size
	
        analysis: list[dict]  = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        res = analysis[0]

        age = res['age']
        gender = res['dominant_gender']
        race = res['dominant_race']
        emotion = res['dominant_emotion']

        print(f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}")
        print(DeepFace.verify(frame, img4_path)['verified'])
        if (DeepFace.verify(frame, img4_path)['verified']): 
            print("same as picture")
        else : 
            print("different from picture")
   

    except Exception as e:
        print("Analysis error:", e)

    # Display the frame using cv2.imshow
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
