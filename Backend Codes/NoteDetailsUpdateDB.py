import sqlite3
import cv2 as cv
import pytesseract
import os.path
import numpy as np


class FiveHFrontUpdateDB:
    def __init__(self, inputImage, file_name, case_no):
        self.filename = file_name
        inputImage = self.preprocess(inputImage)
        serial_num = self.detection(inputImage)
        print(serial_num)
        self.array = list()
        self.CompareFeatures(inputImage, serial_num, file_name, case_no)
        #  Input your create database code here
        cv.waitKey(0)  # needed to view image
        cv.destroyAllWindows()

    # image preprocessinf code
    def preprocess(self, img):
        heightImg, widthImg = img.shape[:2]
        # print(heightImg, widthImg)
        if widthImg >= 2000 & heightImg >= 1000:
            # print('entered ifff')
            img1 = self.resize(img)
            # cv.imshow("resized ", img1)
            cv.waitKey(0)  # needed to view image
            cv.destroyAllWindows()
            return img1
        else:
            img2 = self.cropInput(img)
            img1 = self.resize(img2)
            # cv.imshow("cropped  ", img2)
            cv.waitKey(0)  # needed to view image
            cv.destroyAllWindows()
            # img1 = cv.resize(img2, (1024, 404))
            # cv.imwrite('img.jpg', img1)
            return img1

    def resize(self, img):
        h, w = img.shape[:2]
        # print(h, w)
        ratio = 1784 / w
        dim = (1784, int(h * ratio))
        resize_aspect = cv.resize(img, dim)
        resize = cv.resize(resize_aspect, (1784, 796))
        # cv.imshow('resize', resize)
        # cv.imwrite('1.jpg', resize)
        return resize

    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        return myPointsNew

    def biggestContour(self, contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv.contourArea(i)
            if area > 5000:
                peri = cv.arcLength(i, True)
                approx = cv.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def drawRectangle(self, img, biggest, thickness):
        cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0),
                thickness)
        cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0),
                thickness)
        cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0),
                thickness)
        cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0),
                thickness)

        return img

    def cropInput(self, img):
        heightImg, widthImg = img.shape[:2]  # height, width
        imggrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgblur = cv.GaussianBlur(imggrey, (5, 5), 1)
        imgedge = cv.Canny(imgblur, 30, 75)  # for 2000 note
        # imgedge = cv.Canny(imgblur, 31, 0)         # for 200 note
        kernel = np.ones((5, 5))
        imgdilate = cv.dilate(imgedge, kernel, iterations=2)
        imgerode = cv.erode(imgdilate, kernel, iterations=1)
        imgcontours = img.copy()
        imgbigcontours = img.copy()
        # finding contours
        contours, heirarchy = cv.findContours(imgerode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        draw = cv.drawContours(imgcontours, contours, -1, (0, 255, 0), 10)
        # finding biggest contours
        biggest, maxArea = self.biggestContour(contours)
        if biggest.size != 0:
            biggest = self.reorder(biggest)
            w = biggest[3][0][0]
            h = biggest[2][0][1]
            width = w
            height = h - 20
            # print(width, height)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))
            h, w = imgWarpColored.shape[:2]
            ratio = 1784 / width
            dim = (width, int(height * ratio))
            resize_aspect = cv.resize(imgWarpColored, dim)
            # cv.imshow('img', resize_aspect)
            cv.waitKey()
            return resize_aspect

        elif biggest.size == 0:
            resize = cv.resize(img, (700, 400))
            l_img = np.zeros(shape=[600, 1000, 3], dtype=np.uint8)
            x_offset = y_offset = 50
            l_img[y_offset:y_offset + resize.shape[0], x_offset:x_offset + resize.shape[1]] = resize
            imgbigcontours = resize.copy()
            h, w = img.shape[:2]
            print("resizing")
            ratio = 1784 / w
            dim = (1784, int(h * ratio))
            # dim = (1024, 435)
            resized = cv.resize(imgbigcontours, dim)
            # resized = cv.resize(imgbigcontours, (1024, 456))
            # cv.imshow('dfas', resized)

            cv.waitKey()
            return resized
        else:
            print('cropping failed')

    # serial number detection
    def detection(self, img):
        try:
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            serial_num = cv.imread('Features/500frontfeatures/serial num.jpg')
            # cv.imshow('original', org)
            result = cv.matchTemplate(img, serial_num, cv.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
            x, y = maxLoc
            croppedInput = img[y + 5:y - 13 + 97, x + 10:x - 4 + 461]
            grey = self.toGrayscale(croppedInput)
            threshold_img = cv.threshold(grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_string(threshold_img, lang='eng', config=custom_config)
            data1 = ''
            for i in data:
                if i.isalnum():
                    data1 = data1 + i
            # print(data1)
            data1 = data1.upper()
            if len(data1) == 9:
                return data1
            else:
                print('Enter else- serial number')
                data1 = ''
                croppedInput = img[y:y - 13 + 97, x + 10:x - 4 + 475]
                grey = self.toGrayscale(croppedInput)
                threshold_img = cv.threshold(grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
                # cv.imshow('croppedInput', threshold_img)
                data = pytesseract.image_to_string(threshold_img, lang='eng', config='--psm 6')
                for i in data:
                    if i.isalnum():
                        data1 = data1 + i
                # print(data1)
                data1 = data1.upper()
            return data1
        except AttributeError as err:
            print(err)
        # output = cv.drawKeypoints(grayS, kp, img)
        # return output

    # compare images with already stored template image, return the percentage of similarity
    def templateMatching(self, img1):
        image2 = cv.imread('feature1.jpg')  # original security thread
        result = cv.matchTemplate(image2, img1, cv.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
        x1, y1 = maxLoc
        # print(x1, y1)
        x2, y2 = (12,12)  # your desirable coordinates
        image1 = img1[y1:y2, x1:x2]  # founded area
        image1 = self.toGrayscale(image1)
        image2 = self.toGrayscale(image2)
        ret, image1 = cv.threshold(image1, 127, 255, cv.THRESH_TRUNC)
        ret, image2 = cv.threshold(image2, 127, 255, cv.THRESH_TRUNC)
        result = cv.matchTemplate(image1, image2, cv.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
        maxVal = maxVal * 100
        # print("Security thread =", round(maxVal, 2))
        return round(maxVal, 2)

    def toGrayscale(self, img):
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return imgGray

    def edgeDetection(self, img):
        edges = cv.Canny(img, 100, 250)
        return edges

    def CompareFeatures(self, img1, serial_num, file_name, case_no):
        feature1 = self.templateMatching(img1)
        print(feature1)
        # input your update database code here
