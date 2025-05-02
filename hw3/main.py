from JES import *
from common2 import *
from typing import Callable

myself = Picture.fromjes(makePicture("assets/myself-cr.jpg"))
template = Picture.fromjes(makePicture("assets/template.jpg"))
hand = Picture.fromjes(makePicture("assets/hand.jpg"))
glasses = Picture.fromjes(makePicture("assets/glasses.jpg"))


def whitepoint(threshold: int = 700) -> Callable[[Pixel], None]:
    return lambda px: None if sum(px.color.rgb) < threshold else px.set(Color(255, 255, 255))


def redness(threshold: int, blackness: int = 100) -> Callable[[Pixel], bool]:
    return lambda p: p.color.red - (p.color.green + p.color.blue) >= threshold and sum(p.color.rgb) > blackness


quality = 0.25


def main():  # smallest main function on planet earth
    # deplete quality to make faster
    myself.scale(quality)
    myself.pixmap(whitepoint())
    myself.alpha(Color(255, 255, 255))

    eye1 = (int(546 * quality), int(854 * quality))
    eye2 = (int(880 * quality), int(864 * quality))

    def makeeyesred(px: Pixel):
        nonlocal eye1, eye2

        # with radius 50, scaled by quality
        x1, y1 = eye1
        x2, y2 = px.x, px.y

        x3, y3 = eye2

        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < (50 * quality) ** 2 and sum(px.color.rgb) < 75:
            px.set(Color(255, 0, 0, 200))
        elif (x3 - x2) ** 2 + (y3 - y2) ** 2 < (50 * quality) ** 2 and sum(px.color.rgb) < 75:
            px.set(Color(255, 0, 0, 200))
        elif sum(px.color.rgb) < 150 and px.y <= 800 * quality:
            px.set(Color(128, 0, 128))

        return None

    myself.pixmap(makeeyesred)

    jes = myself.tojes()

    # add text
    addText(jes,
            10, 10, "Done in literally 5 minutes", 16, color=makeColor(0, 255, 0))

    # add rect around text
    addRect(jes, 10, 10, 300, 50, makeColor(0, 255, 0))


main()
