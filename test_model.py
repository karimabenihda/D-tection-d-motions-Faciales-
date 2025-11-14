from tensorflow.keras.models import load_model
import os
import numpy as np
from fastapi.testclient import TestClient
from app.main import app 
def test_model():
    path="./notebook/cnn_saved.keras"
    model=load_model("./notebook/cnn_saved.keras")
    assert os.path.exists(path)
    assert model is not None

client=TestClient(app)

# def test_prediction():

#     model=load_model("./notebook/cnn_saved.keras")
#     img = np.random.rand(1, 48, 48, 3)
#     pred = model.predict(img)
    
#     assert isinstance(pred,np.ndarray)
    

def test_prediction_format():
    with open("notebook/person.jpeg", "rb") as image_file:
        response = client.post("/predict_model", files={"file": image_file})
    assert response.status_code == 200, "La requête a échoué" # Vérifie que la requête a réussi
    data = response.json()
    preds = data["predictions"]
    assert isinstance(preds, list), "'predictions' n'est pas une liste"
    for pred in preds:
        assert "emotion" in pred, "La clé 'emotion' est absente dans une prédiction"
        assert "confidence" in pred, "La clé 'confidence' est absente dans une prédiction"