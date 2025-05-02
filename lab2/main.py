from JES import *


def lab2Fun1(picture):
    for px in getPixels(picture):
        color = getColor(px)
        color = makeDarker(color)
        setColor(px, color)


def lab2Fun2(picture):
    for px in getPixels(picture):
        red = getRed(px)
        green = getGreen(px)
        blue = getBlue(px)
        newColor = makeColor(255-red, 255-green, 255-blue)
        setColor(px, newColor)


def clearRed(picture):
    for px in getPixels(picture):
        [g, b] = [f(px) for f in [getGreen, getBlue]]
        setColor(px, makeColor(0, g, b))


# makes all the pixels in the picture have a blue value of 255
def maxBlue(picture):
    for px in getPixels(picture):
        [r, g] = [f(px) for f in [getRed, getGreen]]
        setColor(px, makeColor(r, g, 255))


# negates green only
def negGreen(picture):
    for px in getPixels(picture):
        [r, g, b] = [f(px) for f in [getRed, getGreen, getBlue]]
        setColor(px, makeColor(r, 255-g, b))


# runs all three
def mangle(picture):
    maxBlue(picture)
    clearRed(picture)
    negGreen(picture)


myPic = makePicture('assets/bibi.jpg')

mangle(myPic)

writePictureTo(myPic, 'output/mangled.jpg')
