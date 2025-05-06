"""
physics.py

This module provides a 2D physics simulation framework. It includes classes for various physics objects
(e.g., circles, ellipses, axis-aligned bounding boxes), a scene to manage these objects, and a writer
to render the simulation. The module also supports collision detection and resolution, as well as
basic physics properties like gravity and restitution.

Dependencies:
- numpy
- PIL (Pillow)
- JES (Jython Environment for Students)
- collisions (custom module for collision handling)
"""

from collections import deque
from JES import *  # type: ignore
from abc import ABC, abstractmethod
import PIL.Image
import numpy as np
import collisions

bb_scene = np.array([500, 500], dtype=np.float64)  # px
background = np.array([42, 42, 42, 255], dtype=np.uint8)
restitution = 0.85  # bounciness of the objects


def blend(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """
    Blend two colors together.

    :param fg: Foreground color (JES color).
    :param bg: Background color (JES color).
    :return: Blended color (JES color).
    """
    rgba1_norm = fg.astype(np.float64) / 255
    rgba2_norm = bg.astype(np.float64) / 255

    a1 = rgba1_norm[3]
    a2 = rgba2_norm[3]

    factor = 1 - a1
    a = a1 + a2 * factor

    if a == 0:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    rgb = (rgba1_norm[:3] * a1 + rgba2_norm[:3] * a2 * factor) / a
    blended_rgba = np.concatenate((rgb, [a])) * 255

    return blended_rgba.astype(np.uint8)


def vec2(x: float, y: float) -> np.ndarray:
    """
    Create a 2D vector.

    :param x: X-coordinate of the vector.
    :param y: Y-coordinate of the vector.
    :return: A 2D numpy array representing the vector.
    """
    return np.array([x, y], dtype=np.float64)


class PhysicsObject(ABC):
    """
    Abstract base class for all physics objects. Defines common properties like position, velocity,
    and mass, as well as abstract methods for drawing and cloning.
    """
    center: np.ndarray
    angle: float

    velocity: np.ndarray
    velocity_angular: float

    mass: float

    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2), angle: float = 0, velocity_angular: float = 0, mass: float = 1.0):
        """
        Initialize the object with position, velocity, and mass.

        :param position: Initial position of the object (2D vector).
        :param velocity: Initial velocity of the object (2D vector).
        :param mass: Mass of the object (float).
        :raises AssertionError: If the position or velocity is not 2D, real, or finite.
        :raises AssertionError: If the mass is not positive.
        :return: None
        """
        assert mass > 0, "mass must be positive"

        self.center = position
        self.velocity = velocity

        self.angle = angle
        self.velocity_angular = velocity_angular

        self.mass = mass

    @abstractmethod
    def draw(self, color: np.ndarray, buf: JESImage):
        """
        Draw the object onto a buffer.

        :param color: Color of the object (JES color).
        :param buf: JESImage buffer to draw onto.
        :return: None
        """
        pass

    @abstractmethod
    def clone(self) -> 'PhysicsObject':
        """
        Create a copy of this object.

        :return: A new instance of the same object.
        """
        pass


class Edge(PhysicsObject):
    """
    Represents the edge of the screen as a physics object. Used for collision handling.
    """

    def __init__(self):
        """
        Initialize the edge object.
        """
        super().__init__(np.zeros(2), np.zeros(2), 0.0, 0.0)

    def draw(self, _: np.ndarray, buf: JESImage):
        """
        Edge objects are not drawn.
        """
        pass

    def clone(self) -> 'Edge':
        """
        Create a copy of this edge.

        :return: A new instance of the same edge.
        """
        return Edge()


class AABB(PhysicsObject):
    """
    Represents an axis-aligned bounding box (AABB) as a physics object.
    """
    box: np.ndarray

    def __init__(self, min: np.ndarray, max: np.ndarray, **kwargs):
        """
        Initialize the AABB object.

        :param min: Bottom left corner.
        :param max: Top right corner.
        :raises AssertionError: If min > max.
        """
        assert np.all(max > min), "max must be greater than min"

        super().__init__(min + 0.5 * (max - min), **kwargs)
        self.box = np.array([min, max], dtype=np.float64)

    @property
    def size(self) -> np.ndarray:
        """
        Get the size of the AABB.

        :return: Size of the AABB (2D vector).
        """
        return self.box[1] - self.box[0]

    @property
    def min(self) -> np.ndarray:
        """
        Get the minimum corner of the AABB, accounting for rotation.

        :return: Minimum corner (2D vector).
        """
        dx, dy = self.center - self.size[0]
        dist = np.hypot(dy, dx)
        theta = np.atan2(dy, dx)
        theta_new = theta + self.angle
        delta_new = dist * np.array([math.cos(theta_new), math.sin(theta_new)])

        return self.center + delta_new

    @property
    def max(self) -> np.ndarray:
        """
        Get the maximum corner of the AABB, accounting for rotation.

        :return: Maximum corner (2D vector).
        """
        dx, dy = self.center - self.size[1]
        dist = np.hypot(dy, dx)
        theta = np.atan2(dy, dx)
        theta_new = theta + self.angle
        delta_new = dist * np.array([math.cos(theta_new), math.sin(theta_new)])

        return self.center - delta_new


class Ellipse(AABB):
    """
    Represents an ellipse as a physics object. Inherits from AABB.
    """

    def __init__(self, center: np.ndarray, radii: np.ndarray, **kwargs):
        """
        Initialize the ellipse with center and radii.

        :param center: Center of the ellipse (2D vector).
        :param radii: Radii of the ellipse (2D vector).
        :raises AssertionError: If the center is not 2D, real, or finite.
        :raises AssertionError: If the radius is not positive.
        :return: None
        """
        super().__init__(center - radii, center + radii, **kwargs)
        assert np.all(radii >= 0), "radii must be positive"

    @property
    def radii(self):
        """
        Get the radii of the ellipse.

        :return: Radii (2D vector).
        """
        return self.size / 2

    @property
    def moi(self):
        """
        Calculate the moment of inertia of the ellipse.

        :return: Moment of inertia (float).
        """
        return self.mass * np.sum(self.radii ** 2) / 4

    def effective_radius(self, dir: np.ndarray | float) -> float:
        """
        Calculate the effective radius of an ellipse at a given angle.

        :param dir: Unit vector [x, y], or theta angle.
        :return: Effective radius (float).
        """
        if not isinstance(dir, np.ndarray):
            dir = np.array([np.cos(dir), np.sin(dir)])

        # -- rotational matrix to local space
        angle = self.angle  # current rotation of the ellipse
        rot_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])

        # -- transform direction into local ellipse space
        local_dir = rot_matrix @ dir
        scaled = local_dir / self.radii

        # -- return the effective radius
        return (1.0 / np.linalg.norm(scaled)).item()

    def draw(self, color: np.ndarray, buf: JESImage):
        """
        Draw the ellipse onto a buffer.

        :param color: Color of the ellipse (JES color).
        :param buf: JESImage buffer to draw onto.
        :return: None
        """
        w, h = (int(x) for x in self.size.astype(np.int32))

        shapebuf = makeEmptyPicture(w, h, color=tuple(background[:3]))

        # -- override the jes image to have a transparency channel, and draw an ellipse
        shapebuf.PILimg = PIL.Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
        d = ImageDraw.Draw(shapebuf.PILimg, mode="RGBA")
        d.ellipse([0, 0, w, h], outline=tuple(color), fill=tuple(color))

        # -- rotate the ellipse
        shapebuf.PILimg = shapebuf.PILimg.rotate(
            np.rad2deg(self.angle),
            expand=1,
            fillcolor=(0, 0, 0, 0)
        )

        # -- overlay the ellipse
        size_shape = np.array([getWidth(shapebuf), getHeight(shapebuf)])
        offx, offy = self.center - size_shape / 2
        copyInto(shapebuf, buf, offx, offy)

        pass

    def clone(self) -> 'Ellipse':
        """
        Create a copy of this ellipse.

        :return: A new instance of the same ellipse.
        """
        return Ellipse(
            self.center.copy(),
            self.radii.copy(),

            velocity=self.velocity.copy(),
            angle=self.angle,
            velocity_angular=self.velocity_angular,
            mass=self.mass
        )


class Circle(Ellipse):
    """
    Represents a circle as a physics object. Inherits from Ellipse.
    """

    def __init__(self, center: np.ndarray, radius: float, **kwargs):
        """
        Initialize the circle with center and radius.

        :param center: Center of the circle (2D vector).
        :param radius: Radius of the circle (float).
        :return: None
        """
        super().__init__(center, np.array([radius, radius]), **kwargs)

    def clone(self) -> 'Circle':
        """
        Create a copy of this circle.

        :return: A new instance of the same circle.
        """
        return Circle(
            self.center.copy(),
            self.radii[0],

            velocity=self.velocity.copy(),
            angle=self.angle,
            velocity_angular=self.velocity_angular,
            mass=self.mass
        )


class CircleTexture(Circle):
    """
    Represents a textured circle as a physics object. Inherits from Circle.
    """
    texture: JESImage

    def __init__(self, center: np.ndarray, radius: float, filename: str, **kwargs):
        """
        Initialize the circle with center, radius, and texture.

        :param center: Center of the circle (2D vector).
        :param radius: Radius of the circle (float).
        :param filename: Filename of the texture (str).
        :raises AssertionError: If the center is not 2D, real, or finite.
        :raises AssertionError: If the radius is not positive.
        :return: None
        """
        super().__init__(center, radius, **kwargs)
        # -- open the image and convert it to RGBA
        texture = JESImage(
            PIL.Image.open(filename).convert("RGBA"),
            filename
        )

        self.texture = texture

    def draw(self, _: np.ndarray, buf: JESImage):
        """
        Draw the textured circle onto a buffer.

        :param _: Ignored color parameter.
        :param buf: JESImage buffer to draw onto.
        :return: None
        """
        tex2 = duplicatePicture(self.texture)

        tex2.PILimg = tex2.PILimg.rotate(
            np.rad2deg(self.angle),
            expand=0,
        )
        tex2.PILimg = tex2.PILimg.resize(
            (int(self.size[0]), int(self.size[0]))
        )

        # -- overlay to buffer
        copyInto(
            tex2,
            buf,
            *(self.center - self.size / 2).astype(np.int32)
        )

    def clone(self) -> 'CircleTexture':
        """
        Create a copy of this textured circle.

        :return: A new instance of the same textured circle.
        """
        return CircleTexture(
            self.center.copy(),
            self.radii[0],

            filename=self.texture.filename,
            velocity=self.velocity.copy(),
            angle=self.angle,
            velocity_angular=self.velocity_angular,
            mass=self.mass
        )


class Scene:
    """
    Represents a 2D scene containing physics objects. Handles updates and collision resolution.
    """
    objects: list[PhysicsObject]
    time_step: float
    gravity: np.ndarray

    def __init__(self, objects: list[PhysicsObject], time_step: float, gravity: np.ndarray = np.zeros(2)):
        """
        Initialize the Scene with a list of objects and a time step.

        :param objects: List of objects in the scene (list of Object).
        :param time_step: Time step for the scene update (float).
        :raises AssertionError: If the time step is not positive.
        :return: None
        """
        assert time_step > 0, "time_step must be positive"

        self.objects = [Edge(), *objects]
        self.time_step = time_step
        self.gravity = gravity.astype(np.float64)

    def update(self) -> None:
        """
        Update the scene by moving the objects according to their velocities and resolving collisions.

        :return: None
        """
        for obj in self.objects:
            if isinstance(obj, Edge):
                continue

            # -- apply gravity
            obj.velocity += self.gravity * self.time_step

            # -- apply velocity
            obj.center += obj.velocity * self.time_step
            obj.angle += obj.velocity_angular * self.time_step

        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i + 1:]:
                collisions.collide(obj1, obj2)


class Writer:
    """
    Handles rendering and writing frames for the physics simulation.
    """
    scene: Scene
    old_objects: list[PhysicsObject]
    frame = 0

    def __init__(self, scene: Scene):
        """
        Initialize the Writer with a scene.

        :param scene: Scene to write to (Scene).
        :return: None
        """
        self.scene = scene
        self.old_objects = []

    def tick(self) -> None:
        """
        Advance the simulation by one time step.

        :return: None
        """
        self.old_objects = [o.clone() for o in self.scene.objects[1:]]
        self.scene.update()

    def draw(self, colors: list[np.ndarray]) -> JESImage:
        """
        Draw the current state of the scene.

        :param colors: List of colors for the objects (list of JES color).
        :return: Rendered JESImage of the scene.
        """
        # -- create an image with alpha channel (!!)
        image = makeEmptyPicture(*bb_scene, color=tuple(background[:3]))
        pil_rgba = PIL.Image.new(
            "RGBA",
            tuple(bb_scene.astype(np.int32)),
            tuple(background)
        )
        image.PILimg = pil_rgba

        # -- dont draw edge obj
        objects = self.scene.objects[1:]

        # -- :)
        if len(colors) != len(objects):
            raise ValueError(
                f"Number of colors ({len(colors)}) must match number of objects ({len(objects)})"
            )

        for obj, color in zip(self.old_objects, colors):
            if isinstance(obj, CircleTexture):
                continue
            color = color.copy() // 2
            obj.draw(color, image)

        for obj, color in zip(self.scene.objects[1:], colors):
            obj.draw(color, image)

        self.frame += 1
        return image
