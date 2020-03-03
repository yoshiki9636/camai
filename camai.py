#! /usr/bin/env python3
# coding: utf-8

import os
import picamera
import RPi.GPIO as GPIO
import time
from time import sleep
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image

# setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_UP)

camera = picamera.PiCamera()
cntr = 0


# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

# Prepare labels.
labels = ReadLabelFile("./categories_jp.txt")
# Initialize engine.
engine = ClassificationEngine("./mobilenet_v2_1.0_224_quant_edgetpu.tflite")

try:
    p = True
    camera.start_preview()
    while p:
        if GPIO.input(23) == GPIO.LOW:
            os.system("./jsay.sh \"かしゃ\"")
            #camera.capture('./image.jpg')
            name = "./image/image" + str(cntr) + ".jpg"
            camera.capture(name)
            cntr = cntr + 1
            img = Image.open(name)
            img2 = img.crop((280,0,720,720))

            start = time.time()
            inf = engine.classify_with_image(img2, top_k=3)
            eltime = time.time() - start
            iftime = engine.get_inference_time()
            print ('num i = ', len(inf))
            if (len(inf) == 0):
                lab2 = u"よくわかりません"
            else:
                for result in inf:
                    print ('---------------------------')
                    print (labels[result[0]])
                    print ('Score : ', result[1])
                print ('elTime : ', eltime)
                print ('ifTime : ', iftime)
                result = inf[0]
                if result[1] >= 0.4:
                    lab2 = u"これは" + labels[result[0]] + "です"
                else:
                    lab2 = u"これはたぶん" + labels[result[0]] + "かもしれません"

            os.system("./jsay.sh \"" + lab2 + "\"")
            print (lab2)

        if GPIO.input(24) == GPIO.LOW:
            p = False
        sleep(0.5)

except KeyboardInterrupt:
    pass

GPIO.cleanup()

print('Finished')
#os.system("sudo shutdown -h now")
