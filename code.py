#experiment 1 Read, Write and Show Images
# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
from google.colab import files

# Upload files
uploaded_files = files.upload()

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import os

image_path = 'goat.jpg'
image1=cv2.imread(image_path)
cv2_imshow(image1)


#experiment 2 Color Space
# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()


image1 = cv2.imread('goat.jpg')
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2_imshow(img)

img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
cv2_imshow(img)

img = cv2.cvtColor(image1, cv2.COLOR_BGR2Lab)
cv2_imshow(img)

img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
cv2_imshow(img)


#expeiment 3 Thresholding Techniques 
# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()

image1 = cv2.imread('goat.png')
ret, thresh1 = cv2.threshold(img, 140, 234, 246, cv2.THRESH_BINARY )
ret, thresh2 = cv2.threshold(img, 45, 247, 234, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 34, 254, 231, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 249, 248, 123, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 237, 167, 189, cv2.THRESH_TOZERO_INV)

cv2_imshow(thresh1) 
cv2_imshow(thresh2) 
cv2_imshow(thresh3) 
cv2_imshow(thresh4) 
cv2_imshow(thresh5)


#experiment 4 Contour Detection
# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()
import numpy as np 

img=cv2.imread('goat.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,160,235,0)
contours,hirearchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print("Number of contours found=",len(contours))
cv2_imshow(img)


#experiment 6 Edge Detection

# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()

img=cv2.imread('goat.jpg')
low=100
upper=250
edge=cv2.Canny(img,low,upper)
cv2_imshow(edge)


#experiment 8 Gaussian Blur, Median Blur & Bilateral 

# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('goat.jpg')
plt.imshow(img)
plt.show()

#size of kernel
gaussian_blur = cv2.GaussianBlur(src=img, ksize=(3,3),sigmaX=0, sigmaY=0)

 #function plt.imshow()
plt.imshow(gaussian_blur)
plt.show()

#Median Blur 


img = cv2.imread('goat.jpg')
plt.imshow(img)
plt.show()
median_blur = cv2.medianBlur(src=img, ksize=9)
plt.show()

# Bilateral Blur

bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2_imshow(bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()



#experiment 11 Face Detection 
# https://github.com/Siw00/CV/blob/main/goat.jpg
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload files
uploaded_files = files.upload()
uploaded_files = files.upload()

img=cv2.imread("goat.jpg")
classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face=classifier.detectMultiScale(gray)
for x,y,w,h in face:
  cv2.rectangle(img,(x,y),(x+w,y+h),(234,45,66),2)

cv2_imshow(img)



#experiment 14 Morphological Erosion & Dilation 

# https://github.com/Siw00/CV/blob/main/goat.jpg
!pip install opencv-python
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
import numpy as np

# Upload files
uploaded_files = files.upload()
img = cv2.imread("goat.jpg")

#defining the kernel matrix
kernel = np.ones((5,5),np.uint8)

erodedimage = cv2.erode(img,kernel,iterations = 1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2_imshow(img_dilation)
cv2.waitKey(0)
