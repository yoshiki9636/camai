#! /usr/bin/env python3
# coding: utf-8

import time
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image

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

name = "./cat.bmp"
img = Image.open(name)
img2 = img.crop((280,0,720,720))
start = time.time()
aaa = engine.classify_with_image(img2, top_k=3)
eltime = time.time() - start
iftime = engine.get_inference_time()

for result in aaa:
    print ('---------------------------')
    print (labels[result[0]])
    print ('Score : ', result[1])
print ('elTime : ', eltime)
print ('ifTime : ', iftime)

