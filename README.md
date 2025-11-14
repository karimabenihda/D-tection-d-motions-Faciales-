# Détection d’Émotions Faciales (CNN + OpenCV + FastAPI + PostgreSQL)

### trello: https://trello.com/b/bTdYjs8q/detection-demotions-faciales


## 📁 Structure du projet:
    Projet/
    │
    ├── .github # Github Action
    │ └── workflows
    │   └── python-tests.yml
    │
    │
    ├── app/ 
    │ ├── .env 
    │ └── main.py
    │
    │
    ├── data/ 
    │ └── test
    │   ├── angry
    |   ├── disgusted
    |   ├── fearful
    |   ├── happy
    |   ├── neutral
    |   ├── sad
    │   └── surprised
    |  
    │ └── train
    │   ├── angry
    |   ├── disgusted
    |   ├── fearful
    |   ├── happy
    |   ├── neutral
    |   ├── sad
    │   └── surprised
    │
    │
    ├── notebook/ 
    │ ├── cnn_saved.keras
    | ├── cnn_saved.h5
    | ├── detect_and_predict.py
    | ├── person.jpeg
    | ├── persons.jpeg
    │ └── train_cnn.ipynb
    │
    │
    ├── .gitignore 
    ├── README.md
    ├── requirements.txt
    └── test_model.py


# 1. EDA:
### Charger le Dataset dataset d’émotions organisé en dossiers nommés par émotion (ex : angry/ ,disgusted/ ,fearful/ ,happy/ ,neutral/ ,sad/ ,surprised) 

### Normalisation (rotation, zoom, flip).



# 2.Train de CNN:
### Créer un CNN avec TensorFlow/Keras avec Couches Conv2D, MaxPooling2D, Flatten, Dense, Dropout.Compiler ,utilisant l'optimiseur adam,et fonction de perte categorical_crossentropy.

### Détection de visages avec OpenCV et Haar Cascade

### Charger le classifieur Haar Cascade 

### Détecter le visage dans une image en entrée avec detectMultiScale().



# 3. Création de l’API FastAPI
### Route POST: /predict_emotion Reçoit un fichier image via UploadFile-> detecte le visge ->returner l'emotion préditet le score 

### Route GET /history :enregistrer l'historique dans la base PostgreSQL.



# 4. Tests unitaires & GitHub Actions
### Vérifier que ton modèle est bien sauvegarde et peut etre recharge sans erreur

### Vérification du format de la prédiction.


