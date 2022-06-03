import glob
import cv2
from PIL import Image
import numpy as np
import os
import pickle


with open("C:/Users/andre/Desktop/hwrt3/IAM-data/IAM-data/iam_lines_gt.txt") as f:
    lines = f.readlines()

lineslist = []
for line in lines:
    lineslist.append(line.strip().split("\t"))
lines = np.array(lineslist)
# print(lineslist)

image_list = []
sentence_list = []
print("Image processing start")
for filename in glob.glob("C:/Users/andre/Desktop/hwrt3/IAM-data/IAM-data/img/*.png"):
    basename = os.path.basename(filename)
    # print(basename)
    for i in range(lines.size):
        if basename == lines[i]:
            sentence_list.append(lines[i + 1])

    im = Image.open(filename)
    im = np.asarray(im)
    image_list.append(im)
image_list = np.array(image_list).tolist()
sentence_list = np.array(sentence_list).tolist()

#print(sentence_list)

list_len = [len(i) for i in sentence_list]
print(max(list_len))


images_to_store = open("images.pickle", "wb")
pickle.dump(image_list, images_to_store)
images_to_store.close()

sentences_to_store = open("sentences.txt", "wb")
pickle.dump(sentence_list, sentences_to_store)
sentences_to_store.close()
