import cv2
import numpy as np
import pygame
from cvzone.HandTrackingModule import HandDetector
import random
import winsound
from time import *
import threading


def countdown():
    global my_timer
    my_timer = 90

    for x in range(my_timer):
        my_timer = my_timer - 1
        sleep(1)


frequency = 800  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second

width = 1280
height = 720
cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

scale = 4
widthKeyboard = 275*scale
heightKeyboard = 72*scale
initialWarpPoints = [[37+240, 271-70], [1197-35, 254-30], [134+50, 481+10], [1107+160, 468+50]]
pts1 = np.float32(initialWarpPoints)
pts2 = np.float32([[0, 0], [widthKeyboard, 0], [0, heightKeyboard], [widthKeyboard, heightKeyboard]])

#variables
isFirstFrame = True
currentKey = 'y'
currentKeyPressed = False
scoreCorrect = 0
scoreWrong = 0
delayCount = 0
result = 0
beepScore = 0
setcnt = 0
once = 1
showtimer = 0
showcharacter = 1
my_timer_initialized = 0

keyLocations = {
    'q': [28-2, 16, 16, 15-4, 'left_pinky'],
    'w': [47-2, 16, 16, 15-4, 'left_ring'],
    'e': [67-2, 16, 16, 15-4, 'left_middle'],
    'r': [86-2, 16, 16, 15-4, 'left_index'],
    't': [105-2, 16, 16, 15-4, 'left_index'],
    'y': [124-2, 16, 16, 15-4, 'right_index'],
    'u': [143-2, 16, 16, 15-4, 'right_index'],
    'i': [162-2, 16, 16, 15-4, 'right_middle'],
    'o': [182-2, 16, 16, 15-4, 'right_ring'],
    'p': [201-2, 16, 16, 15-4, 'right_pinky'],

    'a': [32-1, 19+10, 16, 15-4, 'left_pinky'],
    's': [52-2, 19+10, 16, 15-4, 'left_ring'],
    'd': [71-2, 19+10, 16, 15-4, 'left_middle'],
    'f': [90-2, 19+10, 16, 15-4, 'left_index'],
    'g': [109-2, 19+10, 16, 15-4, 'left_index'],
    'h': [129-2, 19+10, 16, 15-4, 'right_index'],
    'j': [148-2, 19+10, 16, 15-4, 'right_index'],
    'k': [167-2, 19+10, 16, 15-4, 'right_middle'],
    'l': [187-3, 19+10, 16, 15-4, 'right_ring'],

    'z': [42-2, 37+6, 16, 16-4, 'left_pinky'],
    'x': [62-2, 37+6, 16, 16-4, 'left_ring'],
    'c': [81-2, 37+6, 16, 16-4, 'left_middle'],
    'v': [100-2, 37+6, 16, 16-4, 'left_index'],
    'b': [119-2, 37+6, 16, 16-4, 'left_index'],
    'n': [138-2, 37+6, 16, 16-4, 'right_index'],
    'm': [158-2, 37+6, 16, 16-4, 'right_index']
}


pygame.init()

#create Window/Display
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Keyboarding Guide")

#intialize clock for FPS
fps = 30
clock = pygame.time.Clock()

#Creating a dictionary for positions of the keys in the backgroud image
rows = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm']]
rowsStart = [192, 214, 235]
keyLocationsBackground = {}
for i, row in enumerate(rows):
    for j, alphabet in enumerate(row):
        keyLocationsBackground[alphabet] = [rowsStart[i] + 76 * j, 364 + 74 * i, 62, 62]

detector = HandDetector(detectionCon=0.4, maxHands=2)

fingerIds = {
    "8": "index",
    "12": "middle",
    "16": "ring",
    "20": "pinky"
}

# verified

def warpPoint(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    return int(px), int(py)

def checkInside(point, x, y, w, h):
    return x < point[0] < x + w and y < point[1] < y + h



# cnt = -1
while True:
    # cnt += 1
    imgBackground = cv2.imread("Background.png")
    currentKeyPressed = False

    if setcnt == 1:
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()
        showtimer = 1
        setcnt = 0
        my_timer_initialized = 1

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            myKey = getattr(pygame, 'K_{}'.format(currentKey))
            if event.key == myKey:
                print(f'{currentKey} key was pressed')
                currentKeyPressed = True
            # else:
            #     currentKeyPressed = False


    success, img = cap.read()
    if isFirstFrame: cv2.imwrite("Sample.jpg", img)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (widthKeyboard, heightKeyboard))
    imgWarp = cv2.flip(imgWarp, -1)

    hands, img = detector.findHands(img, flipType=True)

    for point in initialWarpPoints:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED) #4 red dots

    if hands:
        # if (my_timer-30) >= 0:
        if ((my_timer_initialized == 1) and ((my_timer - 30) >= 0)) or my_timer_initialized != 1:
            if len(hands) == 2:
                cv2.putText(imgBackground, "Detection: Solid", (50, 50), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)
            else:
                cv2.putText(imgBackground, "Detection: Weak", (50, 50), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 0, 255), 2)

        for hand in hands:
            handType = hand['type'].lower()

            if currentKeyPressed and delayCount == 0:
                key = currentKey
                value = keyLocations[key]
                x, y, w, h = value[0] * scale, value[1] * scale, value[2] * scale, value[3] * scale
                correctFinger = value[4]
                cv2.rectangle(imgWarp, (x, y), (x + w, y + h), (50, 200, 50), cv2.FILLED)

                for id, finger in fingerIds.items():
                    point = hand["lmList"][int(id)]
                    px, py = warpPoint(point, matrix)
                    px = (widthKeyboard - px)  # flip the point
                    py = (heightKeyboard - py)
                    cv2.circle(imgWarp, (px, py), 5, (0, 0, 255), cv2.FILLED)
                    # print(px, py, x, y, w, h)
                    # print(checkInside((px, py), x, y, w, h))



                    if checkInside((px, py), x, y, w, h):
                        # start timer
                        if once == 1:
                            setcnt = 1
                            once = 0

                        print(handType + "_" + finger, correctFinger)
                        if handType + "_" + finger == correctFinger:
                            print("Correct")
                            scoreCorrect += 1
                            currentKey = list(keyLocations)[random.randint(0, 25)]
                            color = [0, 255, 0]
                        else:
                            print("wrong")
                            winsound.Beep(frequency, duration)
                            scoreWrong += 1
                            color = (0, 0, 255)
                        delayCount = 1

                        valueCurrent = keyLocationsBackground[currentKey]
                        x, y, w, h = valueCurrent
                        cv2.rectangle(imgBackground, (x, y), (x + w, y + h), color, cv2.FILLED)
                        cv2.rectangle(imgBackground, (505, 129), (505 + 285, 129 + 177), color, cv2.FILLED)
    else:
        if ((my_timer_initialized == 1) and ((my_timer - 30) >= 0)) or my_timer_initialized != 1:
            cv2.putText(imgBackground, "Detection: Null", (50, 50), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2)
#verified

    for key, value in keyLocations.items():
        x, y, w, h = value[0]*scale, value[1]*scale, value[2]*scale, value[3]*scale
        cv2.rectangle(imgWarp, (x, y), (x+w, y+h), (255, 0, 255), 2)

    cv2.imshow('Image', img)
    cv2.imshow('Image Warp', imgWarp)


    # #Draw on the Backgrond image
    # if currentKeyPressed:
    #Draw all the alphabet on the background image
    for key, val in keyLocationsBackground.items():
        x, y, w, h = val
        cv2.putText(imgBackground, key, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.putText(imgBackground, str(scoreCorrect), (185, 245), cv2.FONT_HERSHEY_PLAIN,
                4, (255, 255, 255), 5)
    cv2.putText(imgBackground, str(scoreWrong), (1035, 245), cv2.FONT_HERSHEY_PLAIN,
                4, (255, 255, 255), 5)

    if showtimer == 1:
        if (my_timer - 30) < 0:
            showcharacter = 0
        if (my_timer-30) >= 0:
            cv2.putText(imgBackground, "timer: ", (910, 70), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2)
            cv2.putText(imgBackground, str(my_timer-30), (1035, 70), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 255), 2)
            cv2.putText(imgBackground, "Speed: ", (540, 70), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2)
            result = round(scoreCorrect / (60 - (my_timer - 30)), 2)
            cv2.putText(imgBackground, str(result), (670, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 2)
        else:
            showcharacter = 0
            beepScore += 1
            if beepScore <= 5:
                frequency = 5000  # Set Frequency To 2500 Hertz
                duration = 500
                winsound.Beep(frequency, duration)
            cv2.putText(imgBackground, "Time Out", (520, 220), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 0, 255), 4)
            # cv2.putText(imgBackground, "timeout", (952, 70), cv2.FONT_HERSHEY_PLAIN,
            #             2, (0, 0, 255), 2)
            cv2.putText(imgBackground, f"Result: {result} char/sec", (370, 60), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 255, 255), 3)
            # cv2.putText(imgBackground, "frequency: ", (500, 70), cv2.FONT_HERSHEY_PLAIN,
            #             2, (0, 255, 0), 2)
            # cv2.putText(imgBackground, str(result), (700, 70), cv2.FONT_HERSHEY_PLAIN,
            #                     2, (0, 255, 0), 2)

        if my_timer == 0:
            break;

    if showcharacter == 1:
        cv2.putText(imgBackground, currentKey, (590, 260), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 255, 255), 20)

    if delayCount != 0:
        delayCount += 1
        if delayCount >= 5:
            delayCount = 0


    #OpenCV Display
    imgRGB = cv2.cvtColor(imgBackground, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    frame = pygame.surfarray.make_surface(imgRGB).convert()
    frame = pygame.transform.flip(frame, True, False)
    window.blit(frame, (0, 0))

    #Update Display
    pygame.display.update()
    #set FPS
    clock.tick(fps)

