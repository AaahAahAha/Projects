import shutil
import os
import random
import glob

os.chdir("C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/data/test")
if os.path.isdir('thumbs_up') is False:
    os.makedirs("thumbs_up")
    os.makedirs("palm")
    os.makedirs("fist")
    os.makedirs("prev")
    os.makedirs("next")
    os.makedirs("swing")
    os.makedirs("one_f")
    os.makedirs("two_f")
    os.makedirs("three_f")
    os.makedirs("none")

    print(len(glob.glob('../train/thumbs_up/*')))#1000-before
    for i in random.sample(glob.glob('../train/thumbs_up/*'), 100):
       shutil.move(i, '../test/thumbs_up')
    for i in random.sample(glob.glob('../train/palm/*'), 100):
        shutil.move(i, '../test/palm')
    for i in random.sample(glob.glob('../train/fist/*'), 100):
        shutil.move(i, '../test/fist')
    for i in random.sample(glob.glob('../train/prev/*'), 100):
        shutil.move(i, '../test/prev')
    for i in random.sample(glob.glob('../train/next/*'), 100):
        shutil.move(i, '../test/next')
    for i in random.sample(glob.glob('../train/swing/*'), 100):
        shutil.move(i, '../test/swing')
    for i in random.sample(glob.glob('../train/one_f/*'), 100):
        shutil.move(i, '../test/one_f')
    for i in random.sample(glob.glob('../train/two_f/*'), 100):
        shutil.move(i, '../test/two_f')
    for i in random.sample(glob.glob('../train/three_f/*'), 100):
        shutil.move(i, '../test/three_f')
    for i in random.sample(glob.glob('../train/none/*'), 100):
        shutil.move(i, '../test/none')

print(len(glob.glob('../train/thumbs_up/*')))#900-after