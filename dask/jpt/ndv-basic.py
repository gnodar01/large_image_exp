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

sys.path.insert(0, os.path.abspath('..'))
import li

# %%
#import importlib
#importlib.reload(li)

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
data_wrapper = CustomDaskWrapper(data_arr)

# %%
viewer = ArrayViewer(data_wrapper, visible_axes=('y', 'x'), channel_axis='c', channel_mode=ChannelMode.COMPOSITE, default_lut={'cmap': Colormap('viridis')})

# %%
viewer.show()

# %%
r = 4
c = 4
chunk_size = 4096
viewer.data = li.data[0][:,chunk_size*r:chunk_size*(r+1),chunk_size*c:chunk_size*(c+1)]

# %%
# 3d
#viewer.display_model.visible_axes = (X,Y,X)
# 2d
#viewer.display_model.visible_axes = (Y,X)

# %%
#viewer.display_model.channel_mode = "grayscale"

# %%
#viewer.display_model.current_index.update({Y: 0})
