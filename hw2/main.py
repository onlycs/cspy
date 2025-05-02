from JES import *
import common


def edge_selector(pic):
    pic2 = duplicatePicture(pic)
    img2d = common.makebetter(pic)

    for i in range(len(img2d)):
        for j in range(len(img2d[0])):
            if common.edge_detection(img2d, i, j):
                setColor(getPixel(pic2, j, i), makeColor(255, 255, 255))
            else:
                setColor(getPixel(pic2, j, i), makeColor(0, 0, 0))

    return pic2


def main():
    pic = makePicture("assets/bibi.jpg")

    p0 = duplicatePicture(pic)
    p1 = edge_selector(common.reflect_y(pic))
    p2 = common.gauss(common.reflect_x(pic), 5)
    p3 = common.reflect_y(common.reflect_x(pic))

    mgyc = [[255, 0, 255], [0, 255, 0], [255, 255, 0], [0, 255, 255]]
    pics = [p0, p1, p2, p3]

    pics = [common.grade(pic, col) for pic, col in zip(pics, mgyc)]
    [p0, p1, p2, p3] = pics

    w = getWidth(pics[0])
    h = getHeight(pics[0])
    pic = []

    for hi in range(2*h):
        for wi in range(2*w):
            if hi < h:
                if wi < w:
                    pic.append(getPixel(p0, wi, hi))
                else:
                    pic.append(getPixel(p1, wi-w, hi))
            else:
                if wi < w:
                    pic.append(getPixel(p2, wi, hi-h))
                else:
                    pic.append(getPixel(p3, wi-w, hi-h))

    px2d = []

    for i in range(0, len(pic), w*2):
        sub = pic[i:i+w*2]
        px2d.append(sub)

    p = makeEmptyPicture(w*2, h*2)

    for i in range(len(px2d)):
        for j in range(len(px2d[i])):
            setColor(getPixel(p, j, i), getColor(px2d[i][j]))

    writePictureTo(p, "test.jpg")


main()
