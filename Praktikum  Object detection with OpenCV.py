#Python
#CopyEdit
import cv2
import numpy as np

# inisialisasi kamera
cap= cv2.VideoCapture(0)

while True:
 frame=cap.read()[1]
 frame= cv2.flip(frame,1)
 hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
 

 # Rentang warna merah dalam HSV 
 lower_red = np.array([100,150,0])
 upper_red = np.array([140,255,255])


 # Masking untuk mendeteksi warna merah
 mask = cv2.inRange(hsv,lower_red, upper_red)
 result = cv2.bitwise_and(frame, frame, mask=mask)
 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 # menggambar bounding box di sekitar objek terdetksi 
 for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

 # Menampilkan hasil
 cv2.imshow("Frame",frame)
 cv2.imshow("Mask",mask)
 cv2.imshow("Result",result)

 if cv2.waitKey(1) & 0xFF == ord('q'):
   break


cap.release()
cv2.destroyAllWindows()
