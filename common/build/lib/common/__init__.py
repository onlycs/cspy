from JES import *
from typing import Callable, TypeAlias

# make it easier to get [r,g,b] from JESPixel
getrgb: list[Callable[[JESPixel], int]] = [getRed, getGreen, getBlue]
setrgb: list[Callable[[JESPixel, int], None]] = [setRed, setGreen, setBlue]

BetterPixel: TypeAlias = list[int]
BetterImage: TypeAlias = list[list[BetterPixel]]


# make a 2d-pixel array
def makebetter(pic: JESImage) -> BetterImage:
    rgbs = jespixels(pic)

    # make 2d-array
    px2d = []  # [[[r, g, b], ...], ...]

    for i in range(0, len(rgbs), getWidth(pic)):
        px2d.append(rgbs[i:i + getWidth(pic)])

    return px2d


# make 1d pixel array
def jespixels(pic: JESImage) -> list[JESPixel]:
    return [[f(p) for f in getrgb] for p in getPixels(pic)]


# flatten 2d-array
def pixels(img: BetterImage) -> list[BetterPixel]:
    return [rgb for row in img for rgb in row]


# assign 1d pixel array to a picture
def assign(pic: JESImage, img1d: list[BetterPixel]):
    for p, rgb in zip(getPixels(pic), img1d):
        setColor(p, makeColor(*rgb))


# clone a 3d array (img2d)
def clone(img2d: BetterImage) -> BetterImage:
    return [[[col for col in rgb] for rgb in row] for row in img2d]


# === GAUSSIAN BLUR === #
def neighbors(img2d: BetterImage, x: int, y: int, rad: int) -> list[BetterPixel]:
    neighbors = []

    for i in range(x - rad, x + rad + 1):
        for j in range(y - rad, y + rad + 1):
            if i >= 0 and i < len(img2d) and j >= 0 and j < len(img2d[0]):
                neighbors.append(img2d[i][j])

    return neighbors


def avg(neighbors) -> BetterPixel:
    rgbtotals = [0 for _ in range(3)]
    rgbcount = 0

    for n in neighbors:
        for i, c in enumerate(n):
            rgbtotals[i] += c
        rgbcount += 1

    return [t // rgbcount for t in rgbtotals]


def gauss(pic: JESImage, rad: int) -> JESImage:
    pic = duplicatePicture(pic)
    img2d = makebetter(pic)  # [[[r, g, b], ...] (row), ... (col)]
    img2d_avg = clone(img2d)

    for x in range(len(img2d)):
        for y in range(len(img2d[0])):
            pxs = neighbors(img2d, x, y, rad)
            gavg = avg(pxs)
            img2d_avg[x][y] = gavg

    for p, rgb in zip(getPixels(pic), pixels(img2d_avg)):
        setColor(p, makeColor(*rgb))

    return pic


# reflect x-axis
def reflect_x(pic: JESImage) -> JESImage:
    pic = duplicatePicture(pic)
    px2d = makebetter(pic)

    # reverse the rows
    rx = px2d[::-1]

    # make 1d
    px = pixels(rx)

    # assign pixels
    assign(pic, px)
    return pic


# reflect y-axis
def reflect_y(pic: JESImage) -> JESImage:
    pic = duplicatePicture(pic)
    px2d = makebetter(pic)

    # reverse the columns
    rx = [row[::-1] for row in px2d]

    # make 1d
    px = pixels(rx)

    # assign pixels
    assign(pic, px)
    return pic


# color grade
def grade(pic: JESImage, rgb: list[int]) -> JESImage:
    pic = duplicatePicture(pic)

    for px in getPixels(pic):
        rgb2 = [f(px) for f in getrgb]
        ave = [(a + b) / 2 for a, b in zip(rgb, rgb2)]
        [f(px, v) for (f, v) in zip(setrgb, ave)]

    return pic


# edge detection
def edge_detection(img2d: BetterImage, x: int, y: int):
    surrounding = neighbors(img2d, x, y, 1)

    if len(surrounding) < 7:
        return False

    left = surrounding[0]
    bottom = surrounding[6]

    return (sum([l - c for l, c in zip(left, bottom)]) / 3) > 6
