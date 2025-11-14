from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@POSTGRES_SERVER:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

with engine.connect() as conn:
    print('connected')
    
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, nullable=False)
    emotion = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app=FastAPI(title="Emotion Detection API")

model=load_model("../notebook/cnn_saved.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.post("/predict_model")
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    if len(faces) == 0:
        return {"error": "Aucun visage detecte dans l'image"}

    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (48, 48))
    face_input = face_resized / 255.0
    face_input = np.expand_dims(face_input, axis=0)
    face_input = np.repeat(face_input[..., np.newaxis], 3, axis=-1)  # Convertir en 3 canaux si necessaire

    # Predict emotion
    predictions = model.predict(face_input)
    emotion_index = int(np.argmax(predictions))
    emotion = emotion_labels[emotion_index]
    score = float(np.max(predictions))

    pred = Prediction(
        image_name=file.filename,
        emotion=emotion,
        score=score
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return {
        "image_name": file.filename,
        "emotion": emotion,
        "score": score
    }
    
    



@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    history=[]
    for r in records:
        history.append({
            "id": r.id,
            "image_name": r.image_name,
            "emotion": r.emotion,
            "score": r.score,
            "created_at": r.created_at.isoformat()
        })
    return {"history": history}
        