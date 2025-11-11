from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from index import face_resized

model_loaded=load_model("cnn_saved.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# img_path="../data/test/happy/im11.png"
# img_path="./persons.jpeg"
# img_path="./rab7a.jpeg"
img_path="./person.jpeg"

img=cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


for (x,y,w,h) in faces:
    face_roi=gray[y:y+h,x:x+w]
    face_resized=cv2.resize(face_roi,(48,48))
    
    face_input = face_resized / 255.0
    face_input = np.expand_dims(face_input, axis=0)  # forme (1,48,48)
    face_input = np.repeat(face_input[..., np.newaxis], 3, axis=-1)  # convertir en 3 canaux

    predictions = model_loaded.predict(face_input)
    emotion_index = np.argmax(predictions)
    emotion = emotion_labels[emotion_index]
    print("Emotion prédite :", emotion)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Détection et prédiction')
plt.show()