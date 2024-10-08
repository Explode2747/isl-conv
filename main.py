# isl conv
import cv2
import mediapipe as mp

mphand = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils

# start webcam to capture frames
vid_cap = cv2.VideoCapture(0)

# detects hands for image something ike that
hands = mphand.Hands()

while vid_cap.isOpened():
    r, frame = vid_cap.read() # to read frames, if not proper breaks loop
    if not r:
        break

    # converting each fram to rgb
    cnvrt_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    procs_hands_onfrm = hands.process(cnvrt_rgb)

    # if hand keymarks/landmarks are present draw over frame
    if procs_hands_onfrm.multi_hand_landmarks:
        for hand_landmarks in procs_hands_onfrm.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame, hand_landmarks, mphand.HAND_CONNECTIONS)
    
    # show video feed
    # overlaying landmarks for showing on videdo capture
    cv2.imshow('landmarks',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid_cap.release()
cv2.destroyAllWindows()

# next steps
# step 1: is to add image input and a database to detect hand sign, traing a numpy dataset for recognition
# step 2: using this trained numpy dataset to recognize signs from feed
# steo 3: output audio or video 