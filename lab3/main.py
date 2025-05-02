from JES import *
from shrinkGrow import *

# mirrors the source picture horizontally
def mirrorHorizontal(source):
  height = getHeight(source)
  mirrorPoint = height // 2
  for x in range(0, getWidth(source)):
    for y in range(mirrorPoint, height):
      topPixel = getPixel(source, x, y)
      bottomPixel = getPixel(source, x, height - y - 1)
      color = getColor(topPixel)
      setColor(bottomPixel, color)

# mirrors the source picture vertically
def mirrorVertical(source):
  width = getWidth(source)
  mirrorPoint = width // 2
  for y in range(0, getHeight(source)):
    for x in range(mirrorPoint, width):
      leftPixel = getPixel(source, x, y)
      rightPixel = getPixel(source, width - x - 1, y)
      color = getColor(leftPixel)
      setColor(rightPixel, color)

# Takes a picture, pic, for input &
# forms a new picture consisting of
# three different sizes of pic
# layered on top of each other.
def layer(pic):
  small = shrink(pic)
  big = grow(pic)
  copyInto(pic, big, (getWidth(big)-getWidth(pic)) // 2, (getHeight(big)-getHeight(pic)) // 2)
  copyInto(small, big, (getWidth(big)-getWidth(small)) // 2, (getHeight(big)-getHeight(small)) // 2)
  writePictureTo(big, "output/layerOutput3.jpg")

def mystery(pic):
  for y in range(0, getHeight(pic)):
    for x in range(0, getWidth(pic) // 2):
      p = getPixel(pic, x, y)
      setRed(p, 0)
      setGreen(p, 255)
      setBlue(p, 255)

def topHalfGreen(pic):
  for y in range(0, getHeight(pic) // 2):
    for x in range(0, getWidth(pic)):
      p = getPixel(pic, x, y)
      setRed(p, 0)
      setGreen(p, 255)
      setBlue(p, 0)

def quad1Yellow(pic):
  for y in range(0, getHeight(pic) // 2):
    for x in range(getWidth(pic) // 2, getWidth(pic)):
      p = getPixel(pic, x, y)
      setRed(p, 255)
      setGreen(p, 255)
      setBlue(p, 0)


# Function definitions above
############################
# Test code below

myPic = makePicture("assets/shops.jpg")
mirrorHorizontal(myPic)
mirrorVertical(myPic)
writePictureTo(myPic, "output/mirrorOutput.jpg")

'''
whitePic = makePicture( "640x480.png" )
TL = getPixel(whitePic, 0, 0)
setBlue(TL, 0)
print(TL)

# Task 1
TR = getPixel(whitePic, getWidth(whitePic) - 1, 0)
setRed(TR, 2)
setGreen(TR, 186)
setBlue(TR, 168)
print(TR)

# Task 2
BR = getPixel(whitePic, getWidth(whitePic) - 1, getHeight(whitePic) - 1)
setRed(BR, 255)
setGreen(BR, 0)
setBlue(BR, 0)
print(BR)

writePictureTo(whitePic, "newPic.png")
'''
