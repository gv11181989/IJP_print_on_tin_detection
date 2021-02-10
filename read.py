import cv2 as cv
import numpy as np
import pytesseract as ocr
from skimage.metrics import structural_similarity as ssim


ocr.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def rescale(src,scale):
    height = int(src.shape[0] * scale)
    width = int(src.shape[1] * scale)
    result = cv.resize(src,(width,height))
    return result


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.mean(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged



img = cv.imread('OCR.jpg')
img = rescale(img,0.4)
# img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
kernel = np.ones((5,3),np.float32)/13
img = cv.filter2D(img,-1,kernel)
d = ocr.image_to_data(img, output_type=ocr.Output.DICT)
d = d['left'][0]
img = img[400:560, 500:1040]




img2 = cv.imread('OCR2.jpg')
img2 = rescale(img2,0.4)
# img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
kernel = np.ones((5,3),np.float32)/13
img2 = cv.filter2D(img2,-1,kernel)
img2 = img2[380:540, 520:1060]




img3 = cv.imread('OCR3.jpg')
img3 = rescale(img3,0.4)
# img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
kernel = np.ones((5,3),np.float32)/13
img3 = cv.filter2D(img3,-1,kernel)
img3 = img3[400:560, 500:1040]




img = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
img2 = cv.fastNlMeansDenoisingColored(img2,None,10,10,7,21)



edge = auto_canny(img,0.66)



kernel2 = np.ones((1,1),np.uint8)#/100
# edge = cv.dilate(edge,kernel2,iterations = 1)
# edge = cv.erode(edge,kernel2,iterations = 1)
edge = np.invert(edge)



edge2 = auto_canny(img2,0.66)
edge2 = np.invert(edge2)


edge3 = auto_canny(img3,0.66)
edge3 = np.invert(edge3)


OCR = ocr.image_to_string(edge)
print(OCR)

OCR = ocr.image_to_string(edge2)
print(OCR)


# d = ocr.image_to_data(edge, output_type=ocr.Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv.rectangle(edge, (x, y), (x + w, y + h), (0, 255, 0), 2)

(score, diff) = ssim(edge, edge2, full=True)
print(score)


(score, diff) = ssim(edge, edge3, full=True)
print(score)




display = np.concatenate((edge, edge2), axis=1)


cv.imshow('output',display)
cv.waitKey(0) 

# capture = cv.VideoCapture('video.mp4')


# while True:
#     isTrue, frame = capture.read()
#     frame = cv.flip(frame, 1)

#     frame = rescale(frame,0.5)

#     edges = cv.Canny(frame,1000,0)

#     cv.imshow('Webcam',edges)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv.destroyAllWindows()