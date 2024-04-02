# from pyfirmata import Arduino, util
import cv2
import dlib
from functools import wraps
from scipy.spatial import distance
import time

# 라즈베리파이에 연결된 부저 울리기
# import RPi.GPIO as GPIO





def calculate_EAR(eye):  # 눈 거리 계산
   A = distance.euclidean(eye[1], eye[5])
   B = distance.euclidean(eye[2], eye[4])
   C = distance.euclidean(eye[0], eye[3])
   ear_aspect_ratio = (A+B)/(2.0*C)
   return ear_aspect_ratio




# 카메라 셋팅
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 940)


# dlib 인식 모델 정의
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
   "shape_predictor_68_face_landmarks.dat")

lastsave = 0
BUZZER_PIN = 7

# 부저 사운드
# def play_sound(duration, freqency):
#     for i in range(duration):
#         GPIO.output(BUZZER_PIN, GPIO.HIGH)
#         time.sleep(freqency)
#         GPIO.output(BUZZER_PIN, GPIO.LOW)
#         time.sleep(freqency)

# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(BUZZER_PIN, GPIO.OUT)
# notes = {'C1': 0.001054,}

# ALARM_ON = False


def counter(func):
   @wraps(func)
   def tmp(*args, **kwargs):
       tmp.count += 1
       time.sleep(0.1)
       global lastsave
       if time.time() - lastsave > 5:
           lastsave = time.time()
           tmp.count = 0
       return func(*args, **kwargs)
   tmp.count = 0
   return tmp




@counter
def close():
   cv2.putText(frame, "Drowsy", (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)



while True:
   ret, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


   faces = hog_face_detector(gray)
   for face in faces:


       face_landmarks = dlib_facelandmark(gray, face)
       leftEye = []
       rightEye = []


       for n in range(36, 42):  # 오른쪽 눈 감지
           x = face_landmarks.part(n).x
           y = face_landmarks.part(n).y
           leftEye.append((x, y))
           next_point = n+1
           if n == 41:
               next_point = 36
           x2 = face_landmarks.part(next_point).x
           y2 = face_landmarks.part(next_point).y
           cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)


       for n in range(42, 48):  # 왼쪽 눈 감지
           x = face_landmarks.part(n).x
           y = face_landmarks.part(n).y
           rightEye.append((x, y))
           next_point = n+1
           if n == 47:
               next_point = 42
           x2 = face_landmarks.part(next_point).x
           y2 = face_landmarks.part(next_point).y
           cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)


       left_ear = calculate_EAR(leftEye)
       right_ear = calculate_EAR(rightEye)


       EAR = (left_ear+right_ear)/2
       EAR = round(EAR, 2)


       if EAR < 0.21:
           close()
           print(f'close count : {close.count}')
           if close.count >= 10:
               print("Drowsy Driving --> Buzzer ON")
            #    for i in range(3):
            #          play_sound(200, notes['C1'])
               time.sleep(0.1)
       print(EAR)


   cv2.imshow("Are you Sleepy", frame)


   key = cv2.waitKey(30)
   if key == 27:
       break


cap.release()
cv2.destroyAllWindows()
