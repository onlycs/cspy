"""
HW5 Animations, by Angad Tendulkar

This script generates animations using physics simulations and 3D transformations.
It includes three parts:
1. Physics with Ellipses
2. Physics with Textures
3. The Third Dimension (3D animations)

The script uses numpy for numerical computations, trimesh for 3D geometry, and JES for image manipulation.
"""

from JES import *  # type:ignore

import text
import physics
from physics import vec2

import numpy as np
import numpy.typing as npt

import shutil
import os

try:
    import trimesh as _
except ImportError:
    print("Trimesh not installed. Installing via pip...")
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "trimesh"])
finally:
    from trimesh.scene.cameras import Camera
    from trimesh.creation import box
    from trimesh import Scene
    from trimesh.transformations import translation_matrix, rotation_matrix

np.set_printoptions(precision=3)
fps = 40


def ellipse(frames: int):
    """
    Simulates and renders an animation of ellipses with random positions, velocities, and angular velocities.

    Args:
        frames (int): The number of frames to render.
    """
    text.set_target("Physics with Ellipses")
    items: list[physics.Ellipse] = []

    for i in range(3):
        pos = np.random.random(2) * 400
        radii = np.random.random(2) * (75 - 40) + 40
        vel = np.random.random(2) * 1000 - 500
        angvel = np.random.rand() * 25 - 15

        # -- ensure x, y not inside another circle
        for j in range(i):
            dx, dy = items[j].center - pos
            dist = np.hypot(dy, dx)

            while dist < np.max(items[j].radii) + np.max(radii):
                pos = np.random.random(2) * 400
                dist = np.linalg.norm(items[j].center - pos).item()

        items.append(physics.Ellipse(
            pos,
            radii,
            velocity=vel,
            velocity_angular=angvel
        ))

    engine = physics.Scene(items, 1 / fps)  # type:ignore
    writer = physics.Writer(engine)

    for i in range(frames):
        writer.tick()
        image = writer.draw([
            np.array([255, 0, 0, 255]),
            np.array([0, 255, 0, 255]),
            np.array([0, 0, 255, 255])
        ])

        image.PILimg = image.PILimg.convert("RGB")
        addText(
            image,
            10, 10,
            text.get_frame(3),
            int(14 * text.get_size()),
            (255, 255, 255)
        )

        writePictureTo(image, f"output/frames/frame{i:04d}.png")

        if i % 10 == 0:
            percent = 100*i/frames
            print(f"[Part 1] Frame {i} of {frames}...\t({percent:.2f}%)")

    print(f"[Part 1] Frame {frames} of {frames}...\t({100:.2f}%)")


def textured(start: int, frames: int):
    """
    Simulates and renders an animation of textured circles with random positions, velocities, and angular velocities.

    Args:
        start (int): The starting frame number.
        frames (int): The number of frames to render.
    """
    text.set_target("Physics with Textures")

    r1 = 50
    r2 = 15
    r3 = 12

    pos1 = np.random.random(2) * 400
    pos2 = np.random.random(2) * 400
    pos3 = np.random.random(2) * 400

    while np.linalg.norm(pos1 - pos2) < r1 + r2:
        pos2 = np.random.random(2) * 400

    while np.linalg.norm(pos1 - pos3) < r1 + r3 or np.linalg.norm(pos2 - pos3) < r2 + r3:
        pos3 = np.random.random(2) * 400

    vel1 = np.random.random(2) * 1000 - 500
    vel2 = np.random.random(2) * 1000 - 500
    vel3 = np.random.random(2) * 1000 - 500

    angvel1 = np.random.rand() * 50 - 25
    angvel2 = np.random.rand() * 50 - 25
    angvel3 = np.random.rand() * 50 - 25

    items = [
        physics.CircleTexture(
            pos1,
            r1,
            "assets/basketball.png",
            velocity=vel1,
            velocity_angular=angvel1
        ),
        physics.CircleTexture(
            pos2,
            r2,
            "assets/baseball.png",
            velocity=vel2,
            velocity_angular=angvel2
        ),
        physics.CircleTexture(
            pos3,
            r3,
            "assets/tennis.png",
            velocity=vel3,
            velocity_angular=angvel3
        )
    ]

    engine = physics.Scene(items, 1 / fps, vec2(0, 980))  # type:ignore
    writer = physics.Writer(engine)
    writer.frame = start

    for i in range(frames):
        writer.tick()
        image = writer.draw([
            np.array([255, 0, 0, 255]),
            np.array([0, 255, 0, 255]),
            np.array([0, 0, 255, 255])
        ])

        image.PILimg = image.PILimg.convert("RGB")
        addText(
            image,
            10, 10,
            text.get_frame(5),
            int(14 * text.get_size()),
            (255, 255, 255)
        )

        writePictureTo(image, f"output/frames/frame{i+start:04d}.png")

        if i % 10 == 0:
            percent = 100*i/frames
            print(f"[Part 2] Frame {i} of {frames}...\t({percent:.2f}%)")

    print(f"[Part 2] Frame {frames} of {frames}...\t({100:.2f}%)")


def dim3anim(start: int):
    """
    Simulates and renders a 3D animation of a cube moving and rotating in space.

    Args:
        start (int): The starting frame number.
    """
    text.set_target("The Third Dimension")

    camera = Camera(resolution=[500, 500], fov=[90, 90])
    camera_current = np.array([0, 0, 650])
    scene = Scene(
        camera=camera,
        camera_transform=translation_matrix(camera_current)
    )

    cube_start = np.array([200., 200., 200.])
    cube_current = cube_start
    cube_target = np.array([-200., -200., 400.])

    cube = box(extents=[100., 100., 100.])
    cube.apply_translation(cube_current)
    scene.add_geometry(cube)

    focal = camera.focal
    res = camera.resolution
    center = np.array(res) / 2

    def ease_in_out(x1: float, x2: float, t: float) -> float:
        """
        Eases a value between two points using a cosine function.

        Args:
            x1 (float): The starting value.
            x2 (float): The ending value.
            t (float): The interpolation factor (0 to 1).

        Returns:
            float: The eased value.
        """
        return x1 + (x2 - x1) * (1 - np.cos(np.pi * t)) / 2

    def t(start: float, end: float, cur: float) -> float:
        """
        Calculates the normalized time factor for interpolation.

        Args:
            start (float): The start time.
            end (float): The end time.
            cur (float): The current time.

        Returns:
            float: The normalized time factor (0 to 1).
        """
        return (cur - start) / (end - start)

    def render(buf: JESImage):
        """
        Renders the current scene to a buffer.

        Args:
            buf (JESImage): The buffer to render to.
        """
        points3 = np.hstack((cube.vertices, np.ones((len(cube.vertices), 1))))
        points2 = (np.linalg.inv(scene.camera_transform) @ points3.T).T[:, :3]
        pixels = focal * (points2[:, :2] / points2[:, 2:3]) + center
        edges = cube.edges_unique

        for edge in edges:
            p1, p2 = pixels[edge]
            addLine(buf, *p1, *p2, color=(255, 255, 255))

        addText(
            buf,
            10, 10,
            text.get_frame(2),
            int(14 * text.get_size()),
            (255, 255, 255)
        )

    def look_at(target: npt.NDArray[np.float64], up=np.array([0, 1, 0])):
        """
        Calculates a camera rotation matrix to look at a target.

        Args:
            target (npt.NDArray[np.float64]): The target position.
            up (npt.NDArray[np.float64]): The up vector.

        Returns:
            npt.NDArray[np.float64]: The rotation matrix.
        """
        forward = camera_current - target
        forward /= np.linalg.norm(forward)

        right = np.cross(up, forward)
        right /= np.linalg.norm(right)

        true_up = np.cross(forward, right)

        rotation = np.eye(4)
        rotation[:3, :3] = np.stack((right, true_up, forward), axis=1)

        return rotation

    def rotate_camera(rotate: npt.NDArray[np.float64]):
        """
        Rotates the camera using a given rotation matrix.

        Args:
            rotate (npt.NDArray[np.float64]): The rotation matrix.
        """
        t = np.eye(4)
        t[:3, 3] = camera_current
        t[:3, :3] = rotate[:3, :3]
        scene.camera_transform = t

    def transform(t1: int, t2: int, p: int):
        """
        Animates the cube's movement and renders frames.

        Args:
            t1 (int): The starting frame number.
            t2 (int): The ending frame number.
            p (int): The part number of the animation.
        """
        nonlocal cube_current

        buf = None
        easing = np.vectorize(ease_in_out)
        for i in range(t1, t2-3):
            next = easing(cube_start, cube_target, t(t1, t2, i))
            diff = next - cube_current
            cube_current = next
            cube.apply_translation(diff)

            rotate_camera(look_at(cube_current))

            buf = makeEmptyPicture(500, 500, color=(20, 20, 20))
            render(buf)
            writePictureTo(buf, f"output/frames/frame{i:04d}.png")

            if i % 10 == 0:
                c = 100*i/(t2)
                print(f"[Part 3.{p}] Frame {t2-i} of {t2-t1}...\t({c:.2f}%)")

        print(f"[Part 3.{p}] Frame {t2-t1} of {t2-t1}...\t(100.00%)")
        writePictureTo(buf, f"output/frames/frame{t2-3:04d}.png")
        writePictureTo(buf, f"output/frames/frame{t2-2:04d}.png")
        writePictureTo(buf, f"output/frames/frame{t2-1:04d}.png")

    # -- do the three transforms
    transform(start, start+50, 1)
    cube_start = cube_target
    cube_target = np.array([-200., -200., -100.])
    transform(start+50, start+100, 2)
    cube_start = cube_target
    cube_target = np.array([-200., -200., 200.])
    transform(start+100, start+150, 3)

    # -- cube rotation
    for i in range(150):
        rotation = rotation_matrix(
            np.deg2rad(5),
            np.array([-200, -200, 0]),
            np.array([-200, -200, 200])
        )
        cube.apply_transform(rotation)

        buf = makeEmptyPicture(500, 500, color=(20, 20, 20))
        render(buf)
        writePictureTo(buf, f"output/frames/frame{start+150+i:04d}.png")

        if i % 10 == 0:
            percent = 100*i/(150)
            print(f"[Part 3.4] Frame {i} of 150...\t({percent:.2f}%)")

    print(f"[Part 3.4] Frame 150 of 150...\t(100.00%)")


shutil.rmtree("output/frames", ignore_errors=True)
os.makedirs("output/frames", exist_ok=True)

ellipse(200)
textured(200, 100)
dim3anim(300)

print(f"[GIF] Concatenating all frames")
movie = makeMovieFromInitialFile("output/frames/frame0000.png")
writeAnimatedGif(movie, "test.gif", fps)
