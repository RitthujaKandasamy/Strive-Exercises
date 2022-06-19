import cv2
import matplotlib.pyplot as plt
import numpy as np


# Define the function that will preprocess the image
def preprocessing(image):
    # Load the image and resize as a square
    img=cv2.imread(image)
    img=cv2.resize(img, (600,600))
    rgb_img=img.copy()

    # Switch the color from BGR to RGB and then from RGB to gray
    rgb_img= cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gray= img.copy()
    gray= cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Set the threshold and find contours
    rtv, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV, +cv2.THRESH_OTSU)    
    contours, hier = cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sort_cont = sorted(contours, key=cv2.contourArea, reverse=True)     # The function computes a contour area.
    drawing = cv2.drawContours(rgb_img, sort_cont[0], -1, (0,255,0), 2)
    
    # Calculate the bounding of each cell
    x,y,w,h = cv2.boundingRect(sort_cont[0])
    new_copy=rgb_img.copy()
    rect=cv2.rectangle(new_copy, (x,y), (x+w, y+h), (0,255,0), 2)

    # Crop the image
    img_crop =threshold[y:y+h, x:x+w]
    img_crop =cv2.resize(img_crop, (1008,1008))

    # Trying with these loops to get all the cells separete.
    for i in range(9):
        for k in range(9):
            for j in range(9):
                image= img_crop[i*111:(i+1)*111, k*111:(k+1)*111]         # tried with several values, the best combination is 111
                f_name= f'cell/cell_{i}{k+1}.png'
                image= cv2.imwrite(f_name,image)

    # plotting the crop img
    plt.imshow(img_crop, cmap='gray')
    plt.show()

    # Plot the result
    # plt.imshow(rect)
    # plt.show()


preprocessing('img/sudoku.png')


