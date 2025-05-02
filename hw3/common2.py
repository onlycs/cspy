from JES import *
from typing import Callable, Self
from abc import ABC, abstractmethod

### Lazy Wrapper for JES ###


class Color:
    # basic dataclass for a color
    # uses white (#fff) as an alpha channel base color
    # because jpegs have no alpha channel
    # alpha is useful for (e.g) OverlayPicture because alpha is necessary there

    rgba: tuple[int, int, int, int]

    def __init__(self, r: int, g: int, b: int, a: int = 255):
        self.rgba = (r, g, b, a)

    @property
    def red(self):
        return self.rgba[0]

    @property
    def green(self):
        return self.rgba[1]

    @property
    def blue(self):
        return self.rgba[2]

    @property
    def alpha(self):
        return self.rgba[3]

    @property
    def rgb(self):
        transparent = [255, 255, 255]
        alphanorm = self.alpha / 255

        return tuple(
            int(alphanorm * self.rgba[i] + (1 - alphanorm) * transparent[i])
            for i in range(3)
        )

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Color) and self.rgba == value.rgba

    def __repr__(self) -> str:
        return "(" + ",".join(list([str(c) for c in self.rgba])) + ")"


class Pixel:
    # a dataclass for a pixel
    # is basically a container for some picture
    # with a reference to x and y coords
    # can make setting and getting known pixels easier

    x: int
    y: int
    ref: 'AnyPicture'

    def __init__(self, picture: 'AnyPicture', x: int, y: int):
        self.ref = picture
        self.x = x
        self.y = y

    @property
    def color(self):
        return self.ref.color(self.x, self.y)

    def set(self, c: Color):
        self.ref.set(self.x, self.y, c)

    def __eq__(self, other):
        return self.color == other.color if isinstance(other, Pixel) else False


# Abstract Class for Any Picture. This is the base class for all the other classes
class AnyPicture(ABC):
    def get(self, x: int, y: int) -> Pixel:
        return Pixel(self, x, y)

    @abstractmethod
    def color(self, x: int, y: int) -> Color:
        pass

    @abstractmethod
    def set(self, x: int, y: int, col: Color):
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @abstractmethod
    def clone(self) -> Self:
        pass

    @abstractmethod
    def tojes(self) -> JESImage:
        pass

    def write(self, path: str):
        writePictureTo(self.tojes(), path)


class MappedPicture(AnyPicture):
    inner: AnyPicture
    f: Callable[[Pixel], Color | None]

    # we need to keep track of which pixels have been mapped.
    mapped: list[list[bool]]

    def __init__(self, inner: AnyPicture, f: Callable[[Pixel], Color | None]):
        self.inner = inner
        self.f = f  # a function that takes in a pixel and either modifies it or returns a new color
        self.mapped = [
            [False for _ in range(self.inner.width)]
            for _ in range(self.inner.height)
        ]

    def color(self, x: int, y: int) -> Color:
        pix = self.inner.get(x, y)
        self.forcemap(x, y)
        return pix.color  # otherwise return the cached value

    def set(self, x: int, y: int, col: Color):
        self.inner.set(x, y, col)
        self.mapped[y][x] = True  # do not map after overwrite

    @property
    def width(self) -> int:
        return self.inner.width

    @property
    def height(self) -> int:
        return self.inner.height

    def clone(self) -> Self:
        return MappedPicture(self.inner.clone(), self.f)  # type: ignore

    def forcemap(self, x: int, y: int):
        if not self.mapped[y][x]:
            self.mapped[y][x] = True
            res = self.f(self.get(x, y))

            if res is not None:
                self.set(x, y, res)

    def tojes(self) -> JESImage:
        [
            self.forcemap(i, j)
            for i in range(self.inner.width)
            for j in range(self.inner.height)
        ]

        return self.inner.tojes()


# Basically an AnyPicture wrapper for JESImage so we can do all this cool lazy stuff
class JESPicture(AnyPicture):
    inner: JESImage
    overrides: dict[tuple[int, int], Color]

    def __init__(self, inner: JESImage, overrides: dict[tuple[int, int], Color] = {}):
        self.inner = inner
        self.overrides = overrides

    def set(self, x: int, y: int, col: Color):
        self.overrides[x, y] = col

    def color(self, x: int, y: int) -> Color:
        if (x, y) in self.overrides:
            return self.overrides[x, y]
        else:
            return Color(*getColor(getPixel(self.inner, x, y)))

    @property
    def width(self) -> int:
        return getWidth(self.inner)

    @property
    def height(self) -> int:
        return getHeight(self.inner)

    def clone(self) -> Self:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(getPixel(pic, i, j), getColor(
                    getPixel(self.inner, i, j)))

        return JESPicture(pic, dict(self.overrides))  # type: ignore

    def apply(self):
        for x, y in self.overrides:
            setColor(getPixel(self.inner, x, y), self.overrides[x, y].rgb)

    def tojes(self) -> JESImage:
        return self.inner


# A picture that is scaled by a factor
class ScaledPicture(AnyPicture):
    inner: AnyPicture
    factor: float
    overrides: dict[tuple[int, int], Color]

    def __init__(self, inner: AnyPicture, factor: float):
        self.inner = inner
        self.factor = factor
        self.overrides = {}

    def color(self, x, y) -> Color:
        if (x, y) in self.overrides:
            return self.overrides[(x, y)]

        return self.inner.color(
            int(x / self.factor),
            int(y / self.factor)
        )

    def set(self, x: int, y: int, col: Color):
        # since we don't store the scaled image (lazy, remember?)
        # we keep a hashmap of the pixels that have been set
        self.overrides[(x, y)] = col

    @property
    def width(self) -> int:
        return int(self.inner.width * self.factor)

    @property
    def height(self) -> int:
        return int(self.inner.height * self.factor)

    def clone(self) -> Self:
        return ScaledPicture(self.inner.clone(), self.factor)  # type: ignore

    def tojes(self) -> JESImage:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(pic, i, j),
                    self.get(i, j).color.rgb  # respect our overrides
                )

        return pic


class OverlayedPicture(AnyPicture):
    back: AnyPicture
    front: AnyPicture
    offset: tuple[int, int]

    def __init__(self, back: AnyPicture, front: AnyPicture, offset: tuple[int, int]):
        self.back = back
        self.front = front
        self.offset = offset

    def color(self, x: int, y: int) -> Color:
        front = self.front.color(x - self.offset[0], y - self.offset[1]) if (
            0 <= x - self.offset[0] < self.front.width and
            0 <= y - self.offset[1] < self.front.height
        ) else None

        back = self.back.color(x, y)

        # sample from the back if we are not using the front
        if not front or front.alpha == 0:
            return back

        if front.alpha == 255:
            return front

        alphanorm = front.alpha / 255

        # if there is alpha, combine the two images
        red = int(alphanorm * front.red + (1 - alphanorm) * back.red)
        green = int(alphanorm * front.green + (1 - alphanorm) * back.green)
        blue = int(alphanorm * front.blue + (1 - alphanorm) * back.blue)

        return Color(red, green, blue, back.alpha)

    def set(self, x: int, y: int, col: Color):
        # set the pixel in the front picture if we are in the overlayed area
        # otherwise set the pixel in the back picture
        self.front.set(x - self.offset[0], y - self.offset[1], col) if (
            0 <= x - self.offset[0] < self.front.width and
            0 <= y - self.offset[1] < self.front.height
        ) else self.back.set(x, y, col)

    @property
    def width(self) -> int:
        return self.back.width

    @property
    def height(self) -> int:
        return self.back.height

    def clone(self) -> Self:
        return OverlayedPicture(
            self.back.clone(),
            self.front.clone(),
            self.offset
        )  # type: ignore

    def tojes(self) -> JESImage:
        for i in range(max(0, self.offset[0]), min(self.width, self.front.width + self.offset[0])):
            for j in range(max(0, self.offset[1]), min(self.height, self.front.height + self.offset[1])):
                self.back.set(i, j, self.color(i, j))

        return self.back.tojes()


# "Functional" class for AnyPicture so we dont have to go around
# instantiating classes and doing python ptr wizardry every time
# we need to do something useful
class Picture(AnyPicture):
    inner: AnyPicture

    def __init__(self, inner: AnyPicture):
        self.inner = inner

    @staticmethod
    def fromjes(jes: JESImage):
        return Picture(JESPicture(jes))

    def pixmap(self, f: Callable[[Pixel], Color | None]):
        self.inner = MappedPicture(self.inner, f)

    def scale(self, factor: float):
        self.inner = ScaledPicture(self.inner, factor)

    def overlay(self, other: AnyPicture, offset: tuple[int, int]):
        self.inner = OverlayedPicture(self.inner, other, offset)

    def alpha(self, p: Color | Callable[[Pixel], bool]):
        pred = p if isinstance(p, Callable) else lambda x: x.color == p
        def map(pix: Pixel): return Color(
            0, 0, 0, 0) if pred(pix) else pix.color

        self.inner = MappedPicture(
            self.inner,
            map
        )

    ### ABC methods ###
    def clone(self) -> Self:
        return Picture(self.inner.clone())  # type: ignore

    def set(self, x: int, y: int, col: Color):
        self.inner.set(x, y, col)

    def color(self, x: int, y: int) -> Color:
        return self.inner.color(x, y)

    @property
    def width(self) -> int:
        return self.inner.width

    @property
    def height(self) -> int:
        return self.inner.height

    def tojes(self) -> JESImage:
        return self.inner.tojes()
