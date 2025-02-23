import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector

from pycocotools import mask as mask_util
from typing import List

import numpy as np
import sys

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        N: the points forming the polygon.
        e.g. [np.array([x1, y1, x2, y2, ...], dtype='float')]

        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0: # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)

    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)

def _maximize(f):
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.full_screen_toggle()

to_int_tuple = lambda t: tuple([int(e) for e in t])

def _onselect(verts):
    global _mask
    global _shape

    flat_verts = []
    for t in verts:
        flat_verts.extend([int(e) for e in t])

    verts = np.array(flat_verts, dtype='float')
    _mask = polygons_to_bitmask([verts], *_shape)

def _on_press(event):
    global _mask
    sys.stdout.flush()
    if event.key == 'enter' and _mask is not None:
        plt.close()

_mask = None
_shape = None
def select_region(image:np.ndarray, height, width):

    img = image.copy()

    global _mask
    global _shape

    _mask = None
    _shape = (height, width)
    fig, ax = plt.subplots()
    _maximize(fig)
    ax.set_xticks([])
    ax.set_yticks([])

    img = img * 255
    img = img.astype('uint8')
    ax.imshow(img)

    fig.canvas.mpl_connect('key_press_event', _on_press)
    rs = PolygonSelector(ax
                        , _onselect
                        , useblit=True
                        , props=dict(color='c', linestyle='--', linewidth=2, alpha=0.5))

    plt.title('Select the illuminated area.', fontsize=22)
    plt.show(block=True)

    return _mask
