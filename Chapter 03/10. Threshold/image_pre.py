import cv2
import matplotlib.pyplot as plt


img = cv2.imread("image\sudoku.png")
print(type(img))

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_img, (7, 7), 3)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)





# plot the image
plt.figure(figsize = (10, 8))
plt.imshow(thresh, cmap= "gray")
plt.show()


