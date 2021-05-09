
import numpy as np
import joblib
import imutils
import cv2
import glob
from googletrans import Translator

ones = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine')

twos = ('Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen')

tens = ('Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety', 'Hundred')

suffixes = ('', 'Thousand', 'Million', 'Billion')


def process(number, index):
    if number == '0':
        return 'Zero'

    length = len(number)

    if (length > 3):
        return False

    number = number.zfill(3)
    words = ''

    hdigit = int(number[0])
    tdigit = int(number[1])
    odigit = int(number[2])

    words += '' if number[0] == '0' else ones[hdigit]
    words += ' Hundred ' if not words == '' else ''

    if (tdigit > 1):
        words += tens[tdigit - 2]
        words += ' '
        words += ones[odigit]

    elif (tdigit == 1):
        words += twos[(int(tdigit + odigit) % 10) - 1]

    elif (tdigit == 0):
        words += ones[odigit]

    if (words.endswith('Zero')):
        words = words[:-len('Zero')]
    else:
        words += ' '

    if (not len(words) == 0):
        words += suffixes[index]

    return words;


def getWords(number):
    length = len(str(number))

    if length > 12:
        return 'This program supports upto 12 digit numbers.'

    count = length // 3 if length % 3 == 0 else length // 3 + 1
    copy = count
    words = []

    for i in range(length - 1, -1, -3):
        words.append(process(str(number)[0 if i - 2 < 0 else i - 2: i + 1], copy - count))
        count -= 1;

    final_words = ''
    for s in reversed(words):
        temp = s + ' '
        final_words += temp

    return final_words

path = glob.glob("*.jpeg")
a=0
for image in path:
    def sort_contours(cnts):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        return (cnts, boundingBoxes)


    b = ""
    img = cv2.imread(image)
    
    img = imutils.resize(img, width=300)

    model = joblib.load('model.pkl')
    
    cv2.imshow("Original", img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Gray Image", gray)

    
    kernel = np.ones((40, 40), np.uint8)

    
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    
    ret, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    
    cnts, hie = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    (cnts, boundingBoxes) = sort_contours(cnts)

    for c in cnts:
        try:
            
            mask = np.zeros(gray.shape, dtype="uint8")

            (x, y, w, h) = cv2.boundingRect(c)

            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)

            
            roi = mask[y - 7:y + h + 7, x - 7:x + w + 7]
            roi = cv2.resize(roi, (28, 28))
            roi = np.array(roi)
            
            roi = roi.reshape(1, 784)

            
            prediction = model.predict(roi)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, str(int(prediction)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
            b = b + str(int(prediction))
        except Exception as e:
            print(e)
    print(b)
    a = a+int(b)
    img = imutils.resize(img, width=500)

    
    cv2.imshow('Detection', img)
    cv2.imwrite(image+".jpeg", img)
print("SUM = "+str(a))
print("IN WORDS = "+getWords(a))
translator = Translator()
result_hindi = translator.translate(getWords(a),dest='hi')
result_german = translator.translate(getWords(a),dest='de')
result_french = translator.translate(getWords(a),dest='fr')
result_russian = translator.translate(getWords(a),dest='ru')
result_arabic = translator.translate(getWords(a),dest='ar')
result_japanese = translator.translate(getWords(a),dest='ja')
result_portuguese = translator.translate(getWords(a),dest='pt')
result_chinese = translator.translate(getWords(a),dest='zh-cn')
result_korean = translator.translate(getWords(a),dest='ko')
print("\nHindi = "+result_hindi.text)
print("\nGerman = "+result_german.text)
print("\nFrench = "+result_french.text)
print("\nRussian = "+result_russian.text)
print("\nArabic = "+result_arabic.text)
print("\nJapanese = "+result_japanese.text)
print("\nPortuguese = "+result_portuguese.text)
print("\nChinese = "+result_chinese.text)
print("\nKorean = "+result_korean.text)

