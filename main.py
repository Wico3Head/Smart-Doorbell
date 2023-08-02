import cv2 as cv
from gpiozero import Button
import pyttsx3
import pygame
pygame.mixer.init()

bell_sound = pygame.mixer.music.load("bell_sound.mp3")
engine = pyttsx3.init()
button = Button(18)

def main():
    motion_detected = False
    button_released = True
    movement_history = [False for i in range(5)]
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

        if button.is_pressed:
            if button_released:
                button_released = False
                if not pygame.mixer.get_busy():
                    pygame.mixer.music.play() 
        elif not button_released:
            button_released = True

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()
