from PIL import Image
import matplotlib.pyplot as plt
import math

im = Image.open("COVID-19_Radiography_Dataset/COVID/images/COVID-11.png", "r")
arr = im.load() #pixel data stored in this 2D array

def rot(A, n, x1, y1): #this is the function which rotates a given block
    temple = []
    for i in range(n):
        temple.append([])
        for j in range(n):
            temple[i].append(arr[x1+i, y1+j])
    for i in range(n):
        for j in range(n):
            arr[x1+i,y1+j] = temple[n-1-i][n-1-j]


xres = 299
yres = 299
BLKSZ = 12 #blocksize
for i in range(2, BLKSZ+1):
    for j in range(int(math.floor(float(xres)/float(i)))):
        for k in range(int(math.floor(float(yres)/float(i)))):
            rot(arr, i, j*i, k*i)
for i in range(3, BLKSZ+1):
    for j in range(int(math.floor(float(xres)/float(BLKSZ+2-i)))):
        for k in range(int(math.floor(float(yres)/float(BLKSZ+2-i)))):
            rot(arr, BLKSZ+2-i, j*(BLKSZ+2-i), k*(BLKSZ+2-i))

plt.figure(0,figsize = (6, 6))
plt.imshow(im,cmap='gray')
plt.show()
