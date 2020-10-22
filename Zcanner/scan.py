from skimage import color
import cv2
import numpy as np
import EndPoint

path = input("paste the path of the image here : ")
image=cv2.imread(path)
orig=image.copy()
cv2.imshow("orignal",image)

image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
gb_image_g =cv2.GaussianBlur(image_g,(3,3),0)
edged=cv2.Canny(gb_image_g,30,50)
contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse=True)


for i in contours:
    j = cv2.arcLength(i, True)
    approx=cv2.approxPolyDP(i,0.02*j,True)

    if len(approx)==4:
        target=approx
        break
        
end_point = EndPoint.point(target)
pts=np.float32([[0,0],[800,0],[800,800],[0,800]])

op=cv2.getPerspectiveTransform(end_point,pts)
dst=cv2.warpPerspective(orig,op,(800,800))

cv2.imshow("Scanned Document",dst)
cv2.imwrite("output.jpg",dst)


cv2.waitKey(0)
cv2.destroyAllWindows()
