import cv2
import numpy as np

# inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()[1]
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Masking untuk mendeteksi warna
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Menggabungkan semua mask
    mask = mask_red | mask_yellow | mask_blue | mask_green
    result = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # menggambar bounding box di sekitar objek terdeteksi
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 255, 0)
            label = "Unknown"
            if cv2.inRange(hsv[y:y+h, x:x+w], lower_red, upper_red).any():
                color = (0, 0, 255)
                label = "Red"
            elif cv2.inRange(hsv[y:y+h, x:x+w], lower_yellow, upper_yellow).any():
                color = (0, 255, 255)
                label = "Yellow"
            elif cv2.inRange(hsv[y:y+h, x:x+w], lower_blue, upper_blue).any():
                color = (255, 0, 0)
                label = "Blue"
            elif cv2.inRange(hsv[y:y+h, x:x+w], lower_green, upper_green).any():
                color = (0, 255, 0)
                label = "Green"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Menampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
