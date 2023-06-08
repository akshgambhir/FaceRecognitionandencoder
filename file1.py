import pickle
import cv2
import face_recognition

# start the video feed
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Load the encoding file or database
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")

# variable declaration
modeType = 0
counter = 0
imageId = -1
imgStudent = []

# compare the image from dataset and display output
while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print("matches", matches)
            # print("faceDis", faceDis)
cv2.imshow("Webcam", img)
cv2.waitKey(1)
