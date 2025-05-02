from JES import *
from typing import Callable, Self
from abc import ABC, abstractmethod

transparent = [255, 255, 255]

### Lazy Wrapper for JES ###


class Color:
    # basic dataclass for a color
    # uses white (#fff) as an alpha channel base color
    # because jpegs have no alpha channel
    # alpha is useful for (e.g) OverlayPicture because alpha is necessary there

    rgba: tuple[int, int, int, float]

    def __init__(self, r: int, g: int, b: int, a: float = 1.0):
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
        global transparent

        return tuple(
            int(self.alpha * self.rgba[i] + (1 - self.alpha) * transparent[i])
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
    def to_jes(self) -> JESImage:
        pass

    def write(self, path: str):
        writePictureTo(self.to_jes(), path)


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
        self.map_n(x, y)
        return self.inner.get(x, y).color

    @property
    def width(self) -> int:
        return self.inner.width

    @property
    def height(self) -> int:
        return self.inner.height

    def clone(self) -> Self:
        return MappedPicture(self.inner.clone(), self.f)  # type: ignore

    def set(self, x: int, y: int, col: Color):
        self.inner.set(x, y, col)
        self.mapped[y][x] = True  # do not map after overwrite

    def map_n(self, x: int, y: int):
        if not self.mapped[y][x]:
            self.mapped[y][x] = True
            res = self.f(self.get(x, y))

            if res is not None:
                self.set(x, y, res)

    def to_jes(self) -> JESImage:
        [
            self.map_n(i, j)
            for i in range(self.inner.width)
            for j in range(self.inner.height)
        ]

        return self.inner.to_jes()


# Basically an AnyPicture wrapper for JESImage so we can do all this cool lazy stuff
class JESPicture(AnyPicture):
    inner: JESImage

    def __init__(self, inner: JESImage):
        self.inner = inner

    def set(self, x: int, y: int, col: Color):
        setColor(getPixel(self.inner, x, y), col.rgb)

    def color(self, x: int, y: int) -> Color:
        return Color(*getColor(getPixel(self.inner, x, y)))

    @property
    def width(self) -> int:
        return getWidth(self.inner)

    @property
    def height(self) -> int:
        return getHeight(self.inner)

    def clone(self) -> Self:
        cloned = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(cloned, i, j),
                    getColor(getPixel(self.inner, i, j))
                )

        return JESPicture(cloned)  # type: ignore

    def to_jes(self) -> JESImage:
        return self.inner


# Readonly Layer
class ReadonlyLayer(AnyPicture):
    inner: AnyPicture
    overrides: dict[tuple[int, int], Color]

    def __init__(self, inner: AnyPicture, overrides: dict[tuple[int, int], Color] = {}):
        self.inner = inner
        self.overrides = overrides

    def color(self, x: int, y: int) -> Color:
        return self.overrides[(x, y)] if (x, y) in self.overrides else self.inner.color(x, y)

    def set(self, x: int, y: int, col: Color):
        self.overrides[(x, y)] = col

    @property
    def width(self) -> int:
        return self.inner.width

    @property
    def height(self) -> int:
        return self.inner.height

    def clone(self) -> Self:
        return ReadonlyLayer(
            self.inner.clone(),
            self.overrides
        )  # type: ignore

    def to_jes(self) -> JESImage:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(pic, i, j),
                    self.get(i, j).color.rgb
                )

        return pic


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

    def to_jes(self) -> JESImage:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(pic, i, j),
                    self.get(i, j).color.rgb  # respect our overrides
                )

        return pic


class OverlayPicture(AnyPicture):
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
        # set the pixel in the front picture if we are in the overlaid area
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
        return OverlayPicture(
            self.back.clone(),
            self.front.clone(),
            self.offset
        )  # type: ignore

    def to_jes(self) -> JESImage:
        for i in range(max(0, self.offset[0]), min(self.width, self.front.width + self.offset[0])):
            for j in range(max(0, self.offset[1]), min(self.height, self.front.height + self.offset[1])):
                self.back.set(i, j, self.color(i, j))

        return self.back.to_jes()


class Shape(AnyPicture, ABC):
    overrides: dict[tuple[int, int], Color]

    @property
    @abstractmethod
    def x(self) -> int:
        pass

    @property
    @abstractmethod
    def y(self) -> int:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def fill(self) -> Color:
        pass

    @property
    @abstractmethod
    def stroke(self) -> Color:
        pass

    @property
    @abstractmethod
    def outline(self) -> bool:
        pass

    @abstractmethod
    def in_stroke(self, x: int, y: int) -> bool:
        pass

    @abstractmethod
    def in_fill(self, x: int, y: int) -> Color:
        pass

    def color(self, x: int, y: int) -> Color:
        return self.fill if (self.in_fill(x, y) and self.outline) or self.in_stroke(x, y) else Color(0, 0, 0, 0)

    def set(self, x: int, y: int, col: Color):
        self.overrides[x, y] = col

    @abstractmethod
    def clone(self) -> Self:
        pass

    def to_jes(self) -> JESImage:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(pic, i, j),
                    self.get(i, j).color.rgb  # respect our overrides
                )

        return pic


class Rectangle(Shape):
    x: int = None  # type: ignore
    y: int = None  # type: ignore
    width: int = None  # type: ignore
    height: int = None  # type: ignore

    outline: bool = None  # type: ignore
    fill: Color = None  # type: ignore
    stroke: Color = None  # type: ignore

    def __init__(self, pos: tuple[int, int], size: tuple[int, int], outline: bool, fill: Color, stroke: Color | None = None):
        self.x, self.y = pos
        self.width, self.height = size
        self.outline = outline
        self.fill = fill
        self.stroke = stroke if stroke is not None else fill

    def in_stroke(self, x: int, y: int) -> bool:
        return x in [self.x, self.x + self.width] or y in [self.y, self.y + self.height]

    def in_fill(self, x: int, y: int) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def clone(self) -> Self:
        return Rectangle(
            (self.x, self.y),
            (self.width, self.height),
            self.outline, self.fill
        )  # type: ignore


class Ellipse(Shape):
    x: int = None  # type: ignore
    y: int = None  # type: ignore
    width: int = None  # type: ignore
    height: int = None  # type: ignore

    outline: bool = None  # type: ignore
    fill: Color = None  # type: ignore
    stroke: Color = None  # type: ignore

    def __init__(self, pos: tuple[int, int], size: tuple[int, int], outline: bool, fill: Color, stroke: Color | None = None):
        self.x, self.y = pos
        self.width, self.height = size
        self.outline = outline
        self.fill = fill
        self.stroke = stroke if stroke is not None else fill

    def in_stroke(self, x: int, y: int) -> bool:
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2

        num_a = (x - cx) ** 2
        num_b = (y - cy) ** 2
        denom_a = (self.width / 2) ** 2
        denom_b = (self.height / 2) ** 2

        return num_a / denom_a + num_b / denom_b == 1

    def in_fill(self, x: int, y: int) -> bool:
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2

        num_a = (x - cx) ** 2
        num_b = (y - cy) ** 2
        denom_a = (self.width / 2) ** 2
        denom_b = (self.height / 2) ** 2

        return num_a / denom_a + num_b / denom_b <= 1

    def clone(self) -> Self:
        return Ellipse(
            (self.x, self.y),
            (self.width, self.height),
            self.outline, self.fill
        )  # type: ignore


class Circle(Ellipse):
    def __init__(self, pos: tuple[int, int], radius: int, outline: bool, fill: Color, stroke: Color | None = None):
        super().__init__(pos, (2 * radius, 2 * radius), outline, fill, stroke)

    def clone(self) -> Self:
        return Circle(
            (self.x, self.y),
            self.width // 2,
            self.outline, self.fill
        )  # type: ignore


class Line(Shape):
    start: tuple[int, int]
    end: tuple[int, int]
    stroke: Color = None  # type: ignore
    fill: Color = Color(0, 0, 0, 0)

    def __init__(self, start: tuple[int, int], end: tuple[int, int], stroke: Color):
        self.start = start
        self.end = end
        self.stroke = stroke

    @property
    def x(self) -> int:
        return min(self.start[0], self.end[0])

    @property
    def y(self) -> int:
        return min(self.start[1], self.end[1])

    @property
    def width(self) -> int:
        return abs(self.start[0] - self.end[0])

    @property
    def height(self) -> int:
        return abs(self.start[1] - self.end[1])

    @property
    def outline(self) -> bool:
        return True

    def in_stroke(self, x: int, y: int) -> bool:
        # y - y2 = (y2-y1/x2-x1)(x-x2)
        x1, y1 = self.start
        x2, y2 = self.end

        return abs(y - y2 - ((y2 - y1) / (x2 - x1)) * (x - x2)) < 50

    def in_fill(self, _x: int, _y: int) -> bool:
        return False

    def clone(self) -> Self:
        return Line(self.start, self.end, self.stroke)  # type: ignore


class Crop(AnyPicture):
    x: int
    y: int
    width: int
    height: int

    inner: AnyPicture

    def __init__(self, inner: AnyPicture, x: int, y: int, width: int, height: int):
        if x + width > inner.width or y + height > inner.height or x < 0 or y < 0 or width < 0 or height < 0:
            raise ValueError("Crop dimensions exceed picture dimensions")

        self.inner = inner
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def color(self, x: int, y: int) -> Color:
        return self.inner.color(x + self.x, y + self.y)

    def set(self, x: int, y: int, col: Color):
        self.inner.set(x + self.x, y + self.y, col)

    def clone(self) -> Self:
        return Crop(
            self.inner.clone(),
            self.x,
            self.y,
            self.width,
            self.height
        )  # type: ignore

    def to_jes(self) -> JESImage:
        pic = makeEmptyPicture(self.width, self.height)

        for i in range(self.width):
            for j in range(self.height):
                setColor(
                    getPixel(pic, i, j),
                    self.get(i, j).color.rgb
                )

        return pic


class Picture(AnyPicture):
    # "Functional" class for AnyPicture so we dont have to go around
    # instantiating classes and doing python ptr wizardry every time
    # we need to do something useful

    inner: AnyPicture

    def __init__(self, inner: AnyPicture):
        if isinstance(inner, Picture):
            self.inner = inner.inner
        else:
            self.inner = inner

    @staticmethod
    def from_jes(jes: JESImage):
        return Picture(ReadonlyLayer(JESPicture(jes)))

    def map(self, f: Callable[[Pixel], Color | None]):
        self.inner = MappedPicture(self.inner, f)

    def scale(self, factor: float):
        self.inner = ScaledPicture(self.inner, factor)

    def overlay(self, other: AnyPicture, offset: tuple[int, int]):
        self.inner = OverlayPicture(self.inner, other, offset)

    def text(self, x: int, y: int, text: str, size: int, w: int, h: int, color: Color = Color(0, 0, 0), background: Color = Color(255, 255, 255)):
        jes = makeEmptyPicture(w, h, background.rgb)
        addText(jes, x, y, text, size, color=color.rgb)

        jes = Picture.from_jes(jes)
        jes.alpha(background)

        self.overlay(jes, (x, y))

    def alpha(self, p: Color | Callable[[Pixel], bool]):
        pred = p if isinstance(p, Callable) else lambda x: x.color == p

        def map(pix: Pixel): return Color(
            0, 0, 0, 0
        ) if pred(pix) else None

        self.inner = MappedPicture(
            self.inner,
            map
        )

    def crop(self, x: int, y: int, width: int, height: int):
        self.inner = Crop(self.inner, x, y, width, height)

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

    def to_jes(self) -> JESImage:
        return self.inner.to_jes()
