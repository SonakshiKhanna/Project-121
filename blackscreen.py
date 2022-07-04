import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,680))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
  ret, bg = cap.read()
bg = np.flip(bg, axis = 1)

while (cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    image = np.flip(image, axis=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])
    mask = cv2.inRange(hsv, u_black, l_black)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    image = cv2.resize(hsv, (640, 480))
    res = cv2.bitwise_and(hsv, hsv, mask = mask)
    f = hsv - res
    f = np.where(f == 0, image, f)

    hsv = cv2.resize(image, (640, 480))    

    final_output = cv2.addWeighted(res)
    output_file.write(final_output)
    
    cv2.imshow("Magic ", final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()