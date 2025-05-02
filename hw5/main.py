import physics
from physics import vec2
import numpy as np
from JES import *  # type:ignore
import shutil
import os
import random
from math import pi

np.set_printoptions(precision=3)
fps = 40


def ellipse(frames: int):
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
        writer.draw([
            np.array([255, 0, 0, 255]),
            np.array([0, 255, 0, 255]),
            np.array([0, 0, 255, 255])
        ])

        if i % 10 == 0:
            percent = 100*i/frames
            print(f"[Part 1] Frame {i} of {frames}...\t({percent:.2f}%)")

    print(f"[Part 1] Frame {frames} of {frames}...\t({100:.2f}%)")


def textured(start: int, frames: int):
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
        writer.draw([
            np.array([255, 0, 0, 255]),
            np.array([0, 255, 0, 255]),
            np.array([0, 0, 255, 255])
        ])

        if i % 10 == 0:
            percent = 100*i/frames
            print(f"[Part 2] Frame {i} of {frames}...\t({percent:.2f}%)")

    print(f"[Part 2] Frame {frames} of {frames}...\t({100:.2f}%)")


shutil.rmtree("output/frames", ignore_errors=True)
os.makedirs("output/frames", exist_ok=True)

ellipse(300)
textured(300, 100)

print(f"[GIF] Concatenating all frames")
movie = makeMovieFromInitialFile("output/frames/frame0000.png")
writeAnimatedGif(movie, "test.gif", fps)
