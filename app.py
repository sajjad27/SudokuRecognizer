#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit to 10MB


@app.route('/recognize-sudoku/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']

    
    if file.filename == '':
        return 'No file selected'
    extension = file.filename.split(".")[-1]
    imagePath = './processed-images/original-image.' + extension

    if file:
        filename = secure_filename(file.filename)
        file.save(imagePath)
        
    sr = Sudonizer()
    result = sr.recognizeSudoku(imagePath)
#         # Perform further processing with the image if needed

    return jsonify(result)

    
    
    
###### real business
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import pytesseract
import imutils
import shutil
import math 
import cv2
import os




class Sudonizer(): 
    

    def recognizeSudoku(self, image_path):
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        self.saveImg(img, 'original image', image_path)
        puzzle, warped = self.find_puzzle(img)
        self.saveImg(puzzle, 'puzzle', image_path)
        squaredImg = self.squareImage(puzzle)
        self.saveImg(squaredImg, 'Squared Image', image_path)
        cellsImages = self.getCellsAsImages(squaredImg)
        self.saveCells(cellsImages, image_path)
        sudokuValues = self.recognize(cellsImages)
        return sudokuValues

        
    
    def find_puzzle(self, image):
        # image pre-processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        # get the external contours and sort them from big to small
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        puzzleCnt = None
        # assume the image itself is the puzzle
        puzzle = image
        warped = gray
        # loop through contours ordered by size until you found contour with 4 corners, 
        # if found then we going to break the loop any ways
        # check  bigger than 50% of the picture, otherwize the image is the puzzly
        for c in cnts:
            height, width = image.shape[:2]
            imgPeri = 2 * width + 2 * height
            contourPeri = cv2.arcLength(c, True)
            contourApprox = cv2.approxPolyDP(c, 0.02 * contourPeri, True)
            if len(contourApprox) == 4:
                contourSizeToImageZiseRatio = ((contourPeri / imgPeri) * 100) 
                # sudoku puzzle should not be smaller than 50% of the image
                differenceThreshold = 50 
                if(contourSizeToImageZiseRatio >= differenceThreshold):
                    newImg = image.copy()
                    cv2.drawContours(newImg, c, -1, (0, 255, 0), 2)
                    puzzle = four_point_transform(image, contourApprox.reshape(4, 2))
#                     h, w = puzzle.shape[:2]
#                     # remove edges
#                     puzzle = puzzle[5:h-5, 6:w-6]
                    warped = four_point_transform(gray, contourApprox.reshape(4, 2))
            break
        
        return (puzzle, warped)

    def squareImage(self, img):
        # Get the dimensions of the input image
        h, w = img.shape[:2]
        # Determine the dimension for making it square
        dim = max(h, w)
        # Resize and stretch the image to fit the square shape with distortion
        stretched_img = cv2.resize(img, (dim, dim))
        return stretched_img
    
    def getCellsAsImages(self, img, num_of_rows=9, num_of_cols=9):
        # get the width and height for the image
        height, width = img.shape[:2]

        part_height = height // num_of_rows
        part_width = width // num_of_cols

        # split each row individually, then for each row split its cells
        cells = {}
        for row_num in range(num_of_cols):
            start_y = row_num * part_height
            end_y = (row_num + 1) * part_height
            row = img[start_y : end_y, :]
            for col_num in range(num_of_rows):
                start_x = col_num * part_width
                end_x = (col_num + 1) * part_width
                cellImg = row[ : , start_x : end_x]

                cellName = str(row_num) + str(col_num)
                cellImg = cv2.fastNlMeansDenoisingColored(cellImg, None, 10, 10, 7, 21)
                cellImg = cv2.cvtColor(cellImg, cv2.COLOR_BGR2GRAY)

                cellImg = cv2.resize(cellImg, (300, 300))
                cellImg = cellImg[40:260, 40:260]

                cells[cellName] = cellImg
                
        return cells
            
   
    def saveCells(self, cells, image_path): 
        cells_path = './processed-images/cells/'
        # Remove the directory and its contents, then recreate it again
        if os.path.exists(cells_path):
            shutil.rmtree(cells_path)
        os.makedirs(cells_path)

        extension = image_path.split(".")[-1]
        for cellName in cells:
            cv2.imwrite(cells_path+cellName+'.'+extension, cells[cellName])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def saveImg(self, img, imgName,image_path, path='./processed-images/'): 
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        extension = image_path.split(".")[-1]
        cv2.imwrite(path+imgName+'.'+extension, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def recognize(self, imgs):
        sudokuValues = {}
        for imgName in imgs:
            cellVal = pytesseract.image_to_string(imgs[imgName], config='--oem 3 --psm 6 -c tessedit_char_whitelist=123456789')
            recognized_number = [int(token) for token in cellVal.split() if token.isdigit()]
            sudokuValues[imgName] = None if not any(recognized_number) else recognized_number[0]
#             sudokuValues[imgName] = recognized_number

        return  sudokuValues



if __name__ == '__main__':
    app.run()


# In[ ]:




