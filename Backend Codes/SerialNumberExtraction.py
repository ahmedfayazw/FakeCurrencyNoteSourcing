import pytesseract


def detection(self, img):
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  #location of pytesseract
        serial_num = cv.imread('serial num.jpg')  #template of serial number image to compare and find coordinates
        result = cv.matchTemplate(img, serial_num, cv.TM_CCOEFF_NORMED)  # algorithm to find the template on input image
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
        x, y = maxLoc  # coordinates of serial number area in input image
        croppedInput = img[y + 5:y - 13 + 97, x + 10:x - 4 + 461]  # cropping the serial number area
        grey = self.toGrayscale(croppedInput)
        threshold_img = cv.threshold(grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]  # applying threshold to seperate the numbers
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_string(threshold_img, lang='eng', config=custom_config)  # algortihm
        data1 = ''
        for i in data:
            if i.isalnum():
                data1 = data1 + i
        data1 = data1.upper()
        if len(data1) == 9:
            return data1
        else:
            print('Enter else- serial number')
            data1 = ''
            croppedInput = img[y:y - 13 + 97, x + 10:x - 4 + 475]
            grey = self.toGrayscale(croppedInput)
            threshold_img = cv.threshold(grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            data = pytesseract.image_to_string(threshold_img, lang='eng', config='--psm 6')
            for i in data:
                if i.isalnum():
                    data1 = data1 + i
            data1 = data1.upper()
        return data1
    except AttributeError as err:
        print(err)
