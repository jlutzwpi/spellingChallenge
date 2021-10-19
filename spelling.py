# program for the little one to practice spelling (and Spanish!)
import cv2
import enum
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image as im
from translate import Translator
from threading import Thread
from datetime import datetime

# only play the spanish word if found and only once
found = False
numFound = 0
time_found = datetime.now()

#play welcome message
os.system('mpg123 sounds/welcome.mp3')

#video stream class for multithreading
class vStream:
    def __init__(self,src,width,height):
        self._running = True
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(src)
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while self._running:
            success,self.frame=self.capture.read()
            if success:
                self.frame2=cv2.resize(self.frame,(self.width,self.height))
    def getFrame(self):
        return self.frame2
    #kill the thread
    def kill(self):
        self.capture.release()
        self._running = False

#play the spanish word if the letter is found
class spanishAudio:
    isFound = False
    fileName = ""

    def __init__(self):
        self._running = True
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while self._running:
            if self.isFound:
                print("Found1")
                cmd = 'mpg123 sounds/' + self.fileName
                os.system(cmd)
                self.isFound = False
    def setFound(self,found, file_path):
        print("Found2")
        self.isFound=found
        self.fileName=file_path
    def kill(self):
        self._running = False

# enumeration of objects to display on the screen
class Object(enum.Enum):
    cat = 1
    dog = 2
    cow = 3
    ball = 4
    duck = 5
    goat = 6

    #increment to the next object
    def inc(self):
        v = self.value + 1
        #if we reached the end, start over
        if v > 6:
            v = 1
        return Object(v)
    #return the missing letter and its position
    #given that the kiddo is just learning letters, only using the first letter
    #set up to have the missing letter be anywhere though
    def letterPos(self):
        l = 1
        if self.value == 1:
            #l = 1
            val = "C"
        if self.value == 2:
            #l = 3
            val = "D"
        if self.value == 3:
            #l = 2
            val = "C"
        if self.value == 4:
            #l = 2
            val = "B"
        if self.value == 5:
            #l = 4
            val = "D"
        if self.value == 6:
            #l = 3
            val = "G"
        return (l,val)

# put cat letters on the screen
def drawCatText(image):
    # show the letters and the one to fill in
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    cv2.putText(image, "A", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "T", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)

    return image

# put duck letters on the screen
def drawDuckText(image):
    # show the letters and the one to fill in
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    #cv2.putText(image, "D", (150, 290),
    #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    cv2.putText(image, "U", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    #image = cv2.rectangle(image, (230, 175), (345, 305), (255, 0, 0), 3)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "C", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)
    cv2.putText(image, "K", (450, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (450, 300), (530, 300), (255, 0, 0), 4)

    return image

# put goat letters on the screen
def drawGoatText(image):
    # show the letters and the one to fill in
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    #cv2.putText(image, "G", (150, 290),
    #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    cv2.putText(image, "O", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "A", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    #image = cv2.rectangle(image, (345, 175), (435, 305), (255, 0, 0), 3)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)
    cv2.putText(image, "T", (450, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (450, 300), (530, 300), (255, 0, 0), 4)

    return image

# put ball letters on the screen
def drawBallText(image):
    # show the letters and the one to fill in
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    #cv2.putText(image, "B", (150, 290),
    #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    cv2.putText(image, "A", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "L", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)
    #image = cv2.rectangle(image, (430, 175), (545, 305), (255, 0, 0), 3)
    cv2.putText(image, "L", (450, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (450, 300), (530, 300), (255, 0, 0), 4)

    return image

# put cow letters on the screen
def drawCowText(image):
    # show the letters and the one to fill in
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    #cv2.putText(image, "C", (150, 290),
    #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    cv2.putText(image, "O", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    #image = cv2.rectangle(image, (230, 175), (345, 305), (255, 0, 0), 3)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "W", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)

    return image

# put dog letters on the screen
def drawDogText(image):
    # show the letters and the one to fill in
    image = cv2.rectangle(image, (130, 175), (245, 305), (255, 0, 0), 3)
    #cv2.putText(image, "D", (150, 290),
    #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (150, 300), (230, 300), (255, 0, 0), 4)
    cv2.putText(image, "O", (250, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    image = cv2.line(image, (250, 300), (330, 300), (255, 0, 0), 4)
    cv2.putText(image, "G", (350, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    #image = cv2.rectangle(image, (345, 175), (440, 305), (255, 0, 0), 3)
    image = cv2.line(image, (350, 300), (430, 300), (255, 0, 0), 4)
    return image

#put the letters on the screen depending on which object it is
def addLetters(curObject, image):
    if curObject.name == "cat":
        image = drawCatText(image)
    elif curObject.name == "dog":
        image = drawDogText(image)
    elif curObject.name == "cow":
        image = drawCowText(image)
    elif curObject.name == "ball":
        image = drawBallText(image)
    elif curObject.name == "duck":
        image = drawDuckText(image)
    elif curObject.name == "goat":
        image = drawGoatText(image)
    return image

# draw the object picture and letters to the screen
def drawScreen(filename, image, curObject):
    game_pic = cv2.imread(filename, 1)
    game_pic = cv2.resize(game_pic, (200, 150), interpolation=cv2.INTER_LINEAR)
    added_image = cv2.addWeighted(
        image[10:160, 200:400, :], 0.1, game_pic[0:150, 0:200, :], 0.9, 0)
    # Change the region with the result
    image[10:160, 200:400] = added_image
    # add the letters for the given object to the screen
    image = addLetters(curObject, image)
    #draw a border around the letters
    image = cv2.rectangle(image, (0, 0), (100, 480), (185, 185, 185), -1)
    image = cv2.rectangle(image, (0, 325), (640, 480), (185, 185, 185), -1)
    image = cv2.rectangle(image, (540, 0), (640, 480), (185, 185, 185), -1)
    return image


# get the input from the screen where the letter goes
def getLetter(image, location):
    get_letter = []
    #only doing the first letter, but can eventually have
    #missing letter anywhere in the word
    get_letter = image[180:298, 130:240]
    #if location == 1:
    #    get_letter = image[180:298, 130:240]
    #if location == 2:
    #    get_letter = image[180:298, 245:335]
    #if location == 3:
    #    get_letter = image[180:298, 345:435]
    #if location == 4:
    #    get_letter = image[180:298, 445:535]
    get_letter = cv2.cvtColor(get_letter, cv2.COLOR_RGB2GRAY)
    get_letter = cv2.resize(get_letter, (28, 28),
                            interpolation=cv2.INTER_LINEAR)
    # invert the black and white colows
    img = cv2.bitwise_not(get_letter)
    # turn the background black
    # if the pixel value is less than 160, that means it's background,
    # so turn it all the way black
    img[img < 160] = 0
    #have dimensions match what goes into the model
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype="float32")
    # rescale image from 0..255 to 0...1
    img /= 255.0

    return img

#tranlate into spanish
def addSpanishWord(curObj, im):
    translator= Translator(to_lang="es")
    translation = translator.translate(curObj.name)
    espanol = "En espanol: " + translation
    cv2.putText(im, espanol, (50, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 8)
    return im

# alphabet labels for the model
dataset_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# load the model
TFLITE_MODEL = "model/model.tflite"
tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
tflite_interpreter.allocate_tensors()

# start with cat
cur_obj = Object.cat


#set width, height, and camera orientation
width = 640
height = 480
flip = 2
camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(
    flip)+' ! video/x-raw, width='+str(width)+', height='+str(height)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#start camera thread
cam1=vStream(camSet,width,height)
#start spanish audio thread
spanish = spanishAudio()
#cap = cv2.VideoCapture(camSet)
# main loop
#while cap.isOpened():
while True:
    #success, image = cap.read()
    try:
        image = cam1.getFrame()
    except:
        print("Frame not found.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # add the picture to spell (we iterate through the enum)
    filename = 'images/' + cur_obj.name + '.jpeg'
    image = drawScreen(filename, image, cur_obj)

    # get the missing letter and run it against the model
    # but have to turn it to grayscale and resize to 28x28 and invert the colors
    (loc,missing_letter) = cur_obj.letterPos()
    img = getLetter(image, loc)

    # Set image into input tensor
    tflite_interpreter.set_tensor(input_details[0]['index'], img)
    # Run inference
    tflite_interpreter.invoke()
    # Get prediction results
    tflite_model_predictions = tflite_interpreter.get_tensor(
        output_details[0]['index'])
    tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
    tflite_pred_dataframe.columns = dataset_labels
    # extract letter from dataframe
    letter_col = tflite_pred_dataframe.iloc[0]
    col_index = letter_col[letter_col > 0.7].index
    if len(col_index) > 0:
        the_letter = col_index[0]
        if the_letter == missing_letter:
            # we found the letter
            found = True
            
    #if the letter was found, play spanish translation
    #put logic here to keep the words on the screen for 3 sec
    #before going to next object
    if found:
        cv2.putText(image, "You found the letter!!!", (100, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

        #translate to spanish and print to the screen
        image = addSpanishWord(cur_obj, image)
        #put a green rectangle around the letter
        image = cv2.rectangle(image, (140, 180), (240, 310), (0, 255, 0), 3)
        #if found for the first time, play the audio
        if numFound == 0:
            sound_str = cur_obj.name + '.mp3'
            spanish.setFound(found,sound_str)
            numFound += 1
            time_found = datetime.now()
        
        # increment to next image after 3 seconds
        if(datetime.now() - time_found).total_seconds() > 3:
            cur_obj = cur_obj.inc()
            #reset found to false and numFound to 0
            found = False
            numFound = 0

    cv2.imshow('Spelling Challenge!', image)
    
    key = cv2.waitKey(1) & 0xFF
    # test capturing the image area to go through model
    if key == ord("c"):
        # capture scaled/converted image
        scaled = img * 255.0
        scaled = np.reshape(scaled, (28, 28))
        data = im.fromarray(scaled)
        data = data.convert("L")
        data.save('bw_img.png')

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
#kill the threads
cam1.kill()
cv2.destroyAllWindows()
#exit(1)