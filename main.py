import cv2 as cv
import pyttsx3

engine = pyttsx3.init()

def main():
    motion_detected = False
    movement_history = [False for i in range(50)]
    mog = cv.createBackgroundSubtractorMOG2()
    cap = cv.VideoCapture(0)

    engine.say("Doorbell is online")
    engine.runAndWait()
    while True:
        ret, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        fgmask = mog.apply(gray)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        fgmask = cv.erode(fgmask, kernel, iterations=1)
        fgmask = cv.dilate(fgmask, kernel, iterations=1)
        
        contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        movement_history.pop(0)
        movement_history.append(len(contours) > 0)

        if not (False in movement_history): 
            if not motion_detected:
                motion_detected = True
                engine.say("Motion Detected")
                engine.runAndWait()
        else:
            motion_detected = False

if __name__ == "__main__":
    main()