# Image Sequence
# 10/1/2024

# This script creates a gif slideshow from a single image.
# It applies a series of filters to the image and saves each
# step as a slide. The slides are then combined into a gif
# slideshow. The filters are as follows:
# 1. Original
# 2. Blue
# 3. Yellow
# 4. White Point
# 5. Black Point
# 6. Up
# 7. Amplify
# 8. Contrast
# 9. Gaussian Blur

from math import floor
from JES import *
import os
from random import random
import colorsys

# make it easier to get [r,g,b] from JESPixel
rgb_fns = [getRed, getGreen, getBlue]

# hardcode src
src = 'assets/bibi.jpg'


# shift array n times
def shift(arr, n):
    return arr[n:] + arr[:n]


# make a 2d-pixel array
def make2d(pic):
    pixels = getPixels(pic)
    rgbs = [[f(p) for f in rgb_fns] for p in pixels]  # [[r, g, b], ...]

    # make 2d-array
    px2d = []  # [[[r, g, b], ...], ...]

    for i in range(0, len(rgbs), getWidth(pic)):
        px2d.append(rgbs[i:i + getWidth(pic)])

    return px2d


# flatten 2d-array
def make1d(px2d):
    return [rgb for row in px2d for rgb in row]


# clone a 3d array (img2d)
def clone(img2d):
    return [[[col for col in rgb] for rgb in row] for row in img2d]


# blue filter
def blue_filter(pic):
    for p in getPixels(pic):
        [r, g, b] = [f(p) for f in rgb_fns]
        setColor(p, makeColor(r, g, b * 2))


# yellow filter
def yellow_filter(pic):
    for p in getPixels(pic):
        [r, g, b] = [f(p) for f in rgb_fns]
        setColor(p, makeColor(r, g, b // 2))


# raise white point
def white_point(pic):
    lim = 200
    for p in getPixels(pic):
        rgb = []

        # raise white point
        for f in rgb_fns:
            c = f(p)
            if c >= lim:
                c = 255
            rgb.append(c)

        setColor(p, makeColor(*rgb))


# raise black point
def black_point(pic):
    lim = 50
    for p in getPixels(pic):
        rgb = []

        # raise black point
        for f in rgb_fns:
            c = f(p)
            if c <= lim:
                c = 0
            rgb.append(c)

        setColor(p, makeColor(*rgb))


# move image up
def up(pic):
    # make 2d-array
    px2d = make2d(pic)

    # move image up (wrap around)
    px2d = shift(px2d, 30)

    # flatten 2d-array
    rgbs = [rgb for row in px2d for rgb in row]

    # set new colors
    for p, rgb in zip(getPixels(pic), rgbs):
        setColor(p, makeColor(*rgb))


# amplify colors
def amplify(pic):
    for p in getPixels(pic):
        rgb = [f(p) for f in rgb_fns]  # [red, green, blue]
        amp = [int(c * 1.6) for c in rgb]  # amplify each color in array
        setColor(p, makeColor(*amp))  # python syntax sugar (array spread)


# contrast increase
def contrast(pic):
    for p in getPixels(pic):
        rgb = [f(p) for f in rgb_fns]  # [red, green, blue]
        cont = [int((c - 128) * 1.2 + 128) for c in rgb]  # stackoverflow fr
        setColor(p, makeColor(*cont))  # python syntax sugar (array spread)


# === GAUSSIAN BLUR === #
def neighbors(img2d, x, y, rad):
    neighbors = []

    for i in range(x - rad, x + rad + 1):
        for j in range(y - rad, y + rad + 1):
            if i >= 0 and i < len(img2d) and j >= 0 and j < len(img2d[0]):
                neighbors.append(img2d[i][j])

    return neighbors


def avg(neighbors):
    rgbtotals = [0 for _ in range(3)]
    rgbcount = 0

    for n in neighbors:
        for i, c in enumerate(n):
            rgbtotals[i] += c
        rgbcount += 1

    return [t // rgbcount for t in rgbtotals]


# gaussian blur
def gauss(pic, rad):
    img2d = make2d(pic)  # [[[r, g, b], ...] (row), ... (col)]
    img2d_avg = clone(img2d)

    for x in range(len(img2d)):
        for y in range(len(img2d[0])):
            pixels = neighbors(img2d, x, y, rad)
            gavg = avg(pixels)
            img2d_avg[x][y] = gavg

    for p, rgb in zip(getPixels(pic), make1d(img2d_avg)):
        setColor(p, makeColor(*rgb))


# === PERLIN NOISE === #
# Whoever came up with this deserves to suffer
class Vec2:
    x: float
    y: float

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def const_of(v):
        h = v & 3

        match h:
            case 0: return Vec2(1, 1)
            case 1: return Vec2(-1, 1)
            case 2: return Vec2(-1, -1)
            case _: return Vec2(1, -1)

    def dot(self, other):
        return self.x * other.x + self.y * other.y


def shuffle(array):
    for i in reversed(range(len(array))):
        repl = round(random() * i-1)
        array[i], array[repl] = array[repl], array[i]


def permutation():
    p = [i for i in range(256)]
    shuffle(p)

    for i in range(256):
        p.append(p[i])

    return p


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(t, a, b):
    return a + t * (b - a)


def noise2d(i, j, perm):
    x = floor(i) & 255
    y = floor(j) & 255

    xf = i - floor(i)
    yf = j - floor(j)

    tr = Vec2(xf-1, yf-1)
    tl = Vec2(xf, yf-1)
    br = Vec2(xf-1, yf)
    bl = Vec2(xf, yf)

    tr_val = perm[perm[x+1] + y+1]
    tl_val = perm[perm[x] + y+1]
    br_val = perm[perm[x+1] + y]
    bl_val = perm[perm[x] + y]

    tr_dot = tr.dot(Vec2.const_of(tr_val))
    tl_dot = tl.dot(Vec2.const_of(tl_val))
    br_dot = br.dot(Vec2.const_of(br_val))
    bl_dot = bl.dot(Vec2.const_of(bl_val))

    u = fade(xf)
    v = fade(yf)

    l1 = lerp(v, bl_dot, tl_dot)
    l2 = lerp(v, br_dot, tr_dot)

    return lerp(u, l1, l2)


def perlin(w, h):
    perlin = [[0 for _ in range(w)] for _ in range(h)]

    rperm = permutation()
    gperm = permutation()
    bperm = permutation()

    for i in range(h):
        for j in range(w):
            # n = (noise2d(float(0.01*i), float(0.01*j), perm) + 1) / 2
            factor = 0.01
            n_red = noise2d(float(factor*i), float(factor*j), rperm)
            n_green = noise2d(float(factor*i), float(factor*j), gperm)
            n_blue = noise2d(float(factor*i), float(factor*j), bperm)

            n = [n_red, n_green, n_blue]
            n = [k + 1 for k in n]
            n = [k / 2 for k in n]

            # for some reason, doing this twice makes it look better
            n = [k + 1 for k in n]
            n = [k / 2 for k in n]

            # convert to rgb
            [r, g, b] = [floor(k*255) for k in n]

            # convert to hsl so i can do meaninful things
            h, l, s = colorsys.rgb_to_hls(r, g, b)

            # bump the saturation
            s *= 4.0
            s = min(s, 1)

            # convert back to rgb
            r, g, b = colorsys.hls_to_rgb(h, l, s)

            perlin[i][j] = [r, g, b]

    return perlin


def edge_detection(img2d, x, y):
    surrounding = neighbors(img2d, x, y, 1)

    if len(surrounding) < 7:
        return False

    left = surrounding[0]
    bottom = surrounding[6]

    return (sum([l - c for l, c in zip(left, bottom)]) / 3) > 6


def perliner(pic):
    img2d = make2d(pic)  # [[[r, g, b], ...] (row), ... (col)]
    img2d_combined = clone(img2d)
    perlinrgb = perlin(getWidth(pic), getHeight(pic))

    for i in range(len(img2d)):
        for j in range(len(img2d[0])):
            if not edge_detection(img2d, i, j):
                img2d_combined[i][j] = perlinrgb[i][j]
            else:
                img2d_combined[i][j] = [100, 100, 100]

    for p, rgb in zip(getPixels(pic), make1d(perlinrgb)):
        setColor(p, makeColor(*rgb))


# make a picture
def makeSlide(f, dest):
    pic = makePicture(src)
    f(pic)
    writePictureTo(pic, dest)


# make slides
def createSequence():
    # create output dir if ne
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # functions to apply for each slide
    fns = [
        lambda x: x,
        blue_filter,
        yellow_filter,
        white_point,
        black_point,
        up,
        amplify,
        contrast,
        lambda x: gauss(x, 10),
        perliner,
    ]

    # make slides
    for i, f in enumerate(fns):
        makeSlide(f, f'tmp/slide0{i + 1}.jpg')

    # cd into output/
    os.chdir('tmp/')

    # make a slideshow
    writeSlideShowTo('../slideshow.gif', 1)

    # cd back
    os.chdir('..')

    # remove tmp dir
    # os.rmdir is not recursive
    # os.system('rm -rf tmp')

    # print width and height
    p = makePicture(src)
    print(f'Width: {getWidth(p)}')
    print(f'Height: {getHeight(p)}')


createSequence()
