from tensorflow.keras.models import load_model
import os
import numpy as np


def test_model():
    path="./notebook/cnn_saved.keras"
    model=load_model("./notebook/cnn_saved.keras")
    assert os.path.exists(path)
    assert model is not None


def test_prediction():
    model=load_model("./notebook/cnn_saved.keras")
    img = np.random.rand(1, 48, 48, 3)
    pred = model.predict(img)
    
    assert isinstance(pred,np.ndarray)