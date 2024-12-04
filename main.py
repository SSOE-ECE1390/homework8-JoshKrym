import easyocr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import digits
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

img = cv2.imread("markers.jpg")
original = cv2.imread("Homework8.jpg")

def getCoordinatesFromCm(pos, w, h):
    paper = (21.59, 27.94)
    print(original.shape)
    proportions = (original.shape[1]/paper[0], original.shape[0]/paper[1])
    actualCM = (pos[0]+paper[0]/2, pos[1]+paper[1]/2)
    x1 = int(actualCM[0]*proportions[0])
    y1 = int(actualCM[1]*proportions[1])
    x2 = int(x1+w*proportions[0])
    y2 = int(y1+h*proportions[1])
    return x1,x2,y1,y2



def cm_to_pixels(cm):
    paper = (21.59, 27.94)
    proportions = (original.shape[0]/paper[0], original.shape[1]/paper[1])
    actualCM = (cm[0]+paper[0]/2, cm[1]+paper[1]/2)
    return actualCM[0]*proportions[0], actualCM[1]*proportions[1]

print(cm_to_pixels((0,0)))
print(cm_to_pixels((-10,-13)))

print(getCoordinatesFromCm((-10,-13),2,10))
x1, x2, y1, y2 = getCoordinatesFromCm((-10,-13),10,2)
plt.imshow(original[int(y1):int(y2),int(x1):int(x2)])
plt.show()

dictionary1=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
corners1,ids1,rejected1 = cv2.aruco.ArucoDetector(dictionary1).detectMarkers(img)

data1 = {}

meansImgData = {}

for idx in range(0,ids1.shape[0]):
    print(ids1[idx])
    thiscorner=corners1[idx][0]
    pos=np.uint16(thiscorner.mean(axis=0))
    meansImgData[ids1[idx][0]] = np.array(pos, dtype=np.float32)
    cv2.putText(img,f"{ids1[idx]}",pos,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

meansImg = [meansImgData[i] for i in sorted(list(meansImgData.keys()))]

print(meansImg)

plt.imshow(img)
plt.show()

dictionary2=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
corners2,ids2,rejected2 = cv2.aruco.ArucoDetector(dictionary2).detectMarkers(original)

data2 = {}

for num, idx in enumerate(ids2):
    data2[idx[0]] = corners2[num]

print(data2)

meansOriginalData = {}

for idx in range(0,ids2.shape[0]):
    print(ids2[idx])
    thiscorner=corners2[idx][0]
    pos=np.uint16(thiscorner.mean(axis=0))
    meansOriginalData[ids2[idx][0]] = np.array(pos, dtype=np.float32)
    #meansOriginal.append(np.array(pos,dtype=np.float32))
    cv2.putText(original,f"{ids2[idx]}",pos,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

meansOriginal = [meansOriginalData[i] for i in sorted(list(meansOriginalData.keys()))]

plt.imshow(original)
plt.show()

p1 = []
for i in corners1:
    for j in i[0]:
        p1.append([j])
p1 = np.array(p1, dtype=np.float32)
print(p1)


p2 = []
for i in corners2:
    for j in i[0]:
        p2.append([j])
p2 = np.array(p2, dtype=np.float32)
print(p2)


print(meansImg)
print(meansOriginal)
#homography = cv2.getAffineTransform(np.array(meansImg, dtype=np.float32), np.array(meansOriginal, dtype=np.float32))
homography = cv2.getAffineTransform(np.array(meansImg, dtype=np.float32), np.array(meansOriginal, dtype=np.float32))
transformed_img = cv2.warpAffine(img,homography, (original.shape[1],original.shape[0]))
plt.imshow(transformed_img)
plt.show()

x1,x2,y1,y2 = getCoordinatesFromCm((-10,-13),10,2)
print(getCoordinatesFromCm((-10,-13),10,2))

#plt.imshow(cv2.threshold(cv2.cvtColor(transformed_img[113:382,147:1449], cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)[1], cmap='gray')
plt.imshow(cv2.cvtColor(transformed_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cmap='gray')
plt.show()


reader = easyocr.Reader(['en'])
result = reader.readtext(transformed_img[y1:y2, x1:x2], paragraph=False)# note that these coordinates, were found by just looking at the original image, rather than doing the pixel math
df=pd.DataFrame(result)
print(df[1])


x1,x2,y1,y2 = getCoordinatesFromCm((-10,-9),2,1)

plt.imshow(cv2.cvtColor(transformed_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), cmap='gray')
plt.show()

print(cv2.cvtColor(transformed_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).mean())

ax1,ax2,ay1,ay2 = getCoordinatesFromCm((-10,-7),2,1)

plt.imshow(cv2.cvtColor(transformed_img[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2GRAY), cmap='gray')
plt.show()

print(f"This student is in {'2390' if (cv2.cvtColor(transformed_img[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2GRAY).mean()<cv2.cvtColor(transformed_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).mean()) else '1390'}")

#top left = 274, 1282
#bottom right = 1262, 1483
amount = int((1262-274)/5)

print(f"amount {amount}")
for i in range(274, 1262, amount):
    print(i)
    print(i+amount)

myTest = [1,2,3,4,5]
print(myTest[1:3:1])

x1,x2,y1,y2 = getCoordinatesFromCm((-9,-4),5*1.5,1.5)

testImage = cv2.cvtColor(transformed_img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
print(amount)
plt.imshow(testImage[:,0:amount])
plt.show()

plt.imshow(cv2.cvtColor(transformed_img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY))
plt.show()

myNumbers = "this"

model = digits.SVM(C=2.67, gamma=5.383)
model.load('digits_svm.dat')
cv2.imwrite("thisDigit.png", testImage[:,amount:2*amount])
plt.imshow(testImage[:,0:amount])
plt.show()
test = digits.preprocess_hog([testImage[:,0:amount], testImage[:,amount:2*amount], testImage[:,2*amount:3*amount], testImage[:,3*amount:4*amount], testImage[:,4*amount:5*amount]])
print(model.predict(test))

#top left = 141, 1934
#bottom right = 1713, 2590

x1,x2,y1,y2 = getCoordinatesFromCm((-10,1),12,5)

testImage = cv2.cvtColor(transformed_img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(testImage,170,255,cv2.THRESH_BINARY)
plt.imshow(thresh1)
plt.show()
cv2.imwrite("tesseract.png", testImage)
plt.imshow(testImage)
plt.show()
#topleft = 1010, 2061
#bottomright = 1086, 2153
text = pytesseract.image_to_string(thresh1[50:-50,50:-50])
plt.imshow(thresh1[50:-50,50:-50])
plt.show()
print(text)
