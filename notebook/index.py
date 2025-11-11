import cv2
import matplotlib.pyplot as plt

# print(image.shape)
# print(image)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

image = cv2.imread("../data/train/angry/im1.png")
# plt.imshow(image)
# plt.show()
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    face_roi=image_gray[y:y+h,x:x+w]
    face_resized=cv2.resize(face_roi,(48,48))
    
    plt.imshow(face_resized,cmap='gray')
    plt.axis('off')
    plt.title('visage redimetionné (48,48)')
    plt.show()


