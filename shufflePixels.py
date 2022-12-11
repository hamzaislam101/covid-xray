from PIL import Image
import matplotlib.pyplot as plt
import math
import os

def rot(A, n, x1, y1): #this is the function which rotates a given block
    temple = []
    for i in range(n):
        temple.append([])
        for j in range(n):
            temple[i].append(A[x1+i, y1+j])
    for i in range(n):
        for j in range(n):
            A[x1+i,y1+j] = temple[n-1-i][n-1-j]


def shuffleImage(path,number):
    im = Image.open(path, "r")
    arr = im.load() #pixel data stored in this 2D array
    xres = 299
    yres = 299
    BLKSZ = 9 #blocksize
    for i in range(2, BLKSZ+1):
        for j in range(int(math.floor(float(xres)/float(i)))):
            for k in range(int(math.floor(float(yres)/float(i)))):
                rot(arr, i, j*i, k*i)
    for i in range(3, BLKSZ+1):
        for j in range(int(math.floor(float(xres)/float(BLKSZ+2-i)))):
            for k in range(int(math.floor(float(yres)/float(BLKSZ+2-i)))):
                rot(arr, BLKSZ+2-i, j*(BLKSZ+2-i), k*(BLKSZ+2-i))

    #plt.figure(0,figsize = (6, 6))
    #plt.imshow(im,cmap='gray')
    im.save("shuffledImages/shuffled-"+str(number)+".png")
    #plt.show()

os.mkdir("shuffledImages")
x = 1
for img in os.listdir("COVID-19_Radiography_Dataset/COVID/images"):
    if x == 100:
        break
    shuffleImage("COVID-19_Radiography_Dataset/COVID/images/" + img,x)
    x += 1
