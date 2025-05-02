from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physics import Ellipse, PhysicsObject


def ellipse_ellipse(e1: Ellipse, e2: Ellipse):
    """
    Check if two circles are colliding. Resolve the collision if they are.
    :param c1: Circle 1
    :param c2: Circle 2
    :return: None
    """

    from physics import restitution

    delta = e2.center - e1.center
    dist = np.linalg.norm(delta)
    dir = np.array([1.0, 0.0]) if dist == 0 else delta / dist

    # -- calculate effective radius in direction of collision
    r1, r2 = e1.effective_radius(dir), e2.effective_radius(-dir)

    # -- mass data unpack
    m1norm = e1.mass / (e1.mass + e2.mass)
    m2norm = e2.mass / (e1.mass + e2.mass)
    m1, m2 = e1.mass, e2.mass

    # -- check if the ellipses are overlapping
    overlap = r1 + r2 - dist

    if overlap < 0:
        return   # no overlap, no collision

    # -- calculate how much to push each ellipse
    e1.center -= dir * overlap * m1norm
    e2.center += dir * overlap * m2norm

    # -- calculate collision point
    contact = (
        (e1.center + dir * r1) +
        (e2.center - dir * r2)
    ) / 2

    rvec1 = contact - e1.center
    rvec2 = contact - e2.center

    # -- calculate velocity at contact point for each ellipse
    v1contact = e1.velocity + np.array([
        -e1.velocity_angular * rvec1[1],
        e1.velocity_angular * rvec1[0]
    ])
    v2contact = e2.velocity + np.array([
        -e2.velocity_angular * rvec2[1],
        e2.velocity_angular * rvec2[0]
    ])

    # -- calculate relative velocity
    v_rel = v2contact - v1contact
    v_norm = np.dot(v_rel, dir)

    if v_norm > 0:
        return

    # -- calculate r cross n
    r1_cross = np.cross(rvec1, dir)
    r2_cross = np.cross(rvec2, dir)

    # -- impulse
    j = (
        (-(1 + restitution) * v_norm) / (
            1 / m1 +
            1 / m2 +
            (r1_cross ** 2)/e1.moi +
            (r2_cross ** 2)/e2.moi
        )
    )

    e1.velocity -= (j / m1) * dir
    e2.velocity += (j / m2) * dir

    e1.velocity_angular -= (j * r1_cross) / e1.moi
    e2.velocity_angular += (j * r2_cross) / e2.moi


def ellipse_edge(e: Ellipse) -> None:
    """
    Check if an ellipse is colliding with an edge. Resolve the collision if it is.
    :param c: Ellipse
    :return: None
    """

    from physics import restitution, bb_scene

    axis_w = e.effective_radius(np.array([1., 0.]))
    axis_h = e.effective_radius(np.array([0., 1.]))
    axis_size = np.array([axis_w, axis_h])
    bb_min = e.center - axis_size
    bb_max = e.center + axis_size

    if np.any(bb_min < 0):
        min_mask = bb_min < 0
        e.center[min_mask] += -bb_min[min_mask]
        e.velocity[min_mask] *= -restitution

    if np.any(bb_max > bb_scene):
        max_mask = bb_max > bb_scene
        e.center[max_mask] += bb_scene[max_mask] - bb_max[max_mask]
        e.velocity[max_mask] *= -restitution


def collide(a: PhysicsObject, b: PhysicsObject):
    from physics import AABB, Edge, Ellipse

    if isinstance(a, Ellipse) and isinstance(b, Ellipse):
        return ellipse_ellipse(a, b)
    if isinstance(a, Ellipse) and isinstance(b, AABB):
        return  # todo
    if isinstance(a, Ellipse) and isinstance(b, Edge):
        return ellipse_edge(a)
    if isinstance(a, AABB) and isinstance(b, AABB):
        return  # todo
    if isinstance(a, AABB) and isinstance(b, Edge):
        return  # todo

    return collide(b, a)
