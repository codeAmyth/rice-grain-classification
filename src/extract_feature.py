import cv2

def extract_features(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours,key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c,True)

    x,y,w,h = cv2.boundingRect(c)

    length = max(w,h)
    width = min(w,h)

    aspect_ratio = length/width

    print("\nPhysical Parameters")
    print("--------------------")
    print("Area:",area)
    print("Perimeter:",perimeter)
    print("Length:",length)
    print("Width:",width)
    print("Aspect Ratio:",aspect_ratio)