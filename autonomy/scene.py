"""Procedural static obstacles for the existing 18-DOF MuJoCo hexapod."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .explorer import Rectangle


@dataclass(frozen=True)
class ExplorationScene:
    xml: str
    obstacles: tuple
    seed: int


def make_scene(base_xml, seed=7, count=5):
    """Keep the existing physics/servos untouched; add only static box geoms."""
    if not 0 <= count <= 12:
        raise ValueError('Expected 0..12 obstacles')
    rng = np.random.default_rng(seed)
    root = ET.fromstring(base_xml)
    world = root.find('worldbody')
    if world is None:
        raise ValueError('MuJoCo XML is missing worldbody')
    boxes = [
        Rectangle(1.35, 0, 0.05, 1.43), Rectangle(-1.35, 0, 0.05, 1.43),
        Rectangle(0, 1.35, 1.43, 0.05), Rectangle(0, -1.35, 1.43, 0.05),
    ]
    added = 0
    for _ in range(500):
        if added >= count:
            break
        x, y = rng.uniform(-0.98, 0.98, 2)
        hx, hy = rng.uniform(0.07, 0.13, 2)
        candidate = Rectangle(float(x), float(y), float(hx), float(hy))
        # Leave enough room for the complete initial foot envelope.
        if np.linalg.norm([x, y]) < 0.51:
            continue
        if any(abs(x-other.x) < hx+other.half_x+0.18 and
               abs(y-other.y) < hy+other.half_y+0.18
               for other in boxes[4:]):
            continue
        boxes.append(candidate)
        added += 1
    if added != count:
        raise RuntimeError('Unable to place the requested obstacle count')
    for i, box in enumerate(boxes):
        ET.SubElement(world, 'geom', {
            'name': f'nav_obstacle_{i:02d}', 'type': 'box',
            'pos': f'{box.x:.6f} {box.y:.6f} {box.half_z:.6f}',
            'size': f'{box.half_x:.6f} {box.half_y:.6f} {box.half_z:.6f}',
            'rgba': '0.67 0.34 0.17 1',
            'friction': '0.9 0.01 0.0001', 'condim': '4',
        })
    return ExplorationScene(ET.tostring(root, encoding='unicode'), tuple(boxes), int(seed))
