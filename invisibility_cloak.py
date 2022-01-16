import cv2
import numpy as np

cap = cv2.VideoCapture(0)


for i in range(60):
    ret,background = cap.read()
    background = cv2.resize(background, ( 1280,960))
	#check if the frame is returned then brake
background=np.flip(background, axis=1)

while cap.isOpened():
    ret, img=  cap.read()
    img = cv2.resize(img, (1280,960))
    
    img=np.flip(img, axis=1)
    
    if(ret):
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
        # lower red range
        lower=np.array([0,120,50])
        upper=np.array([10,255,255])
        mask1=cv2.inRange(hsv, lower, upper)
           
        #    upper red range
        l_red = np.array([170,120,70])
        u_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv, l_red, u_red)

        #generating the final red mask
        red_mask = mask1 + mask2

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 5) 
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1)  
    
        part1 = cv2.bitwise_and(background, background, mask= red_mask)

        red_free = cv2.bitwise_not(red_mask)

        # if cloak is not present show the current image
        part2 = cv2.bitwise_and(img, img, mask= red_free)
        
        cv2.imshow('cloak', part1 + part2)
        if cv2.waitKey(1) & 0xFF == ord('q'):     
            break 
    
    
cap.release()
cv2.destroyAllWindows()