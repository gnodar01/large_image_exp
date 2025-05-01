# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
from ndv import ArrayViewer
from ndv.models import DataWrapper, ChannelMode
import numpy as np
from cmap import Colormap
from typing import Any, TypeGuard, Hashable, Mapping, Sequence
import dask.array.core as da
import ipywidgets as widgets

sys.path.insert(0, os.path.abspath('..'))
import li

# %%
#import importlib
#importlib.reload(li)

# %%
dirr = lambda x: [a for a in dir(x) if not a.startswith('_')]

# %%
li.data

# %%
data_arr = li.data[-1]

# zcyx
data_arr.shape, data_arr.dtype, data_arr.min(), data_arr.max()

# %%
# label to idx
LI = {
 'c': 0,
 'y': 1,
 'x': 2,
}
# index to label
IL = dict((v,k) for k,v in LI.items())

C = LI['c']
Y = LI['y']
X = LI['x']

# %%
class CustomWrapper(DataWrapper):
    PRIORITY = 10

    __li = {
     'c': 0,
     'y': 1,
     'x': 2,
    }

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[np.ndarray]:
        return isinstance(obj, np.ndarray)

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self.__li.keys())

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        return {label: range(self._data.shape[idx]) for label, idx in self.__li.items()}

class CustomDaskWrapper(DataWrapper):
    PRIORITY = 10

    __li = {
     'c': 0,
     'y': 1,
     'x': 2,
    }

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[da.Array]:
        if (da := sys.modules.get("dask.array")) and isinstance(obj, da.Array):
            return True
        return False

    def _asarray(self, data: da.Array) -> np.ndarray:
        return np.asarray(data.compute())

    def save_as_zarr(self, path: str) -> None:
        self._data.to_zarr(url=path)

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self.__li.keys())

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        return {label: range(self._data.shape[idx]) for label, idx in self.__li.items()}

# %%
standard_luts = [
    {'visible': True, 'cmap': Colormap('red')},
    {'visible': True, 'cmap': Colormap('green')},
    {'visible': True, 'cmap': Colormap('blue')},
]

num_visible_axes = min(data_arr.shape[C], len(standard_luts))
visible_axes = list(range(num_visible_axes))
luts = {ax: standard_luts[ax] for ax in visible_axes}
for ax in range(num_visible_axes, data_arr.shape[C]):
    luts[ax] = {'visible': False}

# %%
viewer = ArrayViewer(
    CustomDaskWrapper(li.data[-1]),
    visible_axes=('y', 'x'),
    channel_axis='c',
    channel_mode=ChannelMode.COMPOSITE,
    default_lut={'visible': False, 'cmap': Colormap('viridis')},
    luts=luts
)

# %%
widget_output = widgets.Output()

def set_data(nth, lvl):
    tile = li.tile_n(nth=nth, frame=slice(0,data_arr.shape[C],1), level=lvl, do_transpose=False)
    viewer.data = tile

def level_change(change):
    with widget_output:
        nth = tile_slider.value
        lvl = change['new']
        lvl = max(0, lvl)
        lvl = min(lvl, len(li.data)-1)
        tile_slider.max = li.nn(lvl)-1
        set_data(nth, lvl)

def tile_change(change):
    with widget_output:
        lvl = level_slider.value
        max_nth = li.nn(lvl)
        nth = change['new']
        nth = max(0, nth)
        nth = min(nth, max_nth)
        set_data(nth, lvl)
        

level_slider = widgets.IntSlider(min=0, max=len(li.data)-1, value=len(li.data)-1, description="res")
level_slider.observe(level_change, names="value")

tile_slider = widgets.IntSlider(min=0, max=0, value=0, description="tile")
tile_slider.observe(tile_change, names="value")


# %%
def go_level_up(b):
    with widget_output:
        level_slider.value -= 1
def go_level_down(b):
    with widget_output:
        print("level down from", level_slider.value, b)
        level_slider.value += 1

def go_tile_up(b):
    with widget_output:
        old_nth = tile_slider.value
        level = level_slider.value
        new_nth = old_nth - li.nx(level)
        if new_nth >= 0:
            tile_slider.value = new_nth
def go_tile_down(b):
    with widget_output:
        old_nth = tile_slider.value
        level = level_slider.value
        new_nth = old_nth + li.nx(level)
        if new_nth < li.nn(level):
            tile_slider.value = new_nth
def go_tile_left(b):
    with widget_output:
        old_nth = tile_slider.value
        level = level_slider.value
        old_x = old_nth % li.nx(level)
        if old_x > 0:
            tile_slider.value -= 1
def go_tile_right(b):
    with widget_output:
        old_nth = tile_slider.value
        level = level_slider.value
        old_x = old_nth % li.nx(level)
        if old_x < (li.nx(level) -1):
            tile_slider.value += 1

btn = lambda icon: widgets.Button(
    description='',
    disabled=False,
    button_style='',
    tooltip='go level up',
    icon=icon
)

level_up_btn = btn('arrow-up')
level_up_btn.on_click(go_level_up)
level_down_btn = btn('arrow-down')
level_down_btn.on_click(go_level_down)


tile_up_btn = btn('arrow-up')
tile_up_btn.on_click(go_tile_up)
tile_down_btn = btn('arrow-down')
tile_down_btn.on_click(go_tile_down)
tile_left_btn = btn('arrow-left')
tile_left_btn.on_click(go_tile_left)
tile_right_btn = btn('arrow-right')
tile_right_btn.on_click(go_tile_right)

# %%
display(level_slider)

display(level_up_btn, level_down_btn)

display(tile_slider)

display(tile_up_btn, tile_down_btn, tile_left_btn, tile_right_btn)

viewer.show()

# %%
for i in range(3, data_arr.shape[C]):
    viewer.display_model.luts[i].visible = True

# %%
r = 4
c = 4
chunk_size = 4096
viewer.data = li.data[0][:,chunk_size*r:chunk_size*(r+1),chunk_size*c:chunk_size*(c+1)]

# %%
canvas = viewer._canvas
do_zoom = lambda z: canvas.set_range(margin=-(2**z-1))

# %%
do_zoom(0)

# %%
# 3d
#viewer.display_model.visible_axes = (X,Y,X)
# 2d
#viewer.display_model.visible_axes = (Y,X)

# %%
#viewer.display_model.channel_mode = "grayscale"

# %%
#viewer.display_model.current_index.update({Y: 0})

# %%
