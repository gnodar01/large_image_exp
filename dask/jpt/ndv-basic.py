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
from ndv import data, ArrayViewer
from ndv.models import DataWrapper, ChannelMode
import numpy as np
from cmap import Colormap
from typing import Any, TypeGuard, Hashable, Mapping, Sequence
from .. import li

# %%
data_arr = data.cells3d()

# %%
# zcyx
data_arr.shape, data_arr.dtype, data_arr.min(), data_arr.max()

# %%
# label to idx
LI = {
 'z': 0,
 'c': 1,
 'y': 2,
 'x': 3,
}
# index to label
IL = dict((v,k) for k,v in LI.items())

Z = LI['z']
C = LI['c']
Y = LI['y']
X = LI['x']

# %%
class CustomWrapper(DataWrapper):
    PRIORITY = 10

    __li = {
     'z': 0,
     'c': 1,
     'y': 2,
     'x': 3,
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

# %%
data_wrapper = CustomWrapper(data_arr)

# %%
viewer = ArrayViewer(data_wrapper, visible_axes=('y', 'x'), channel_axis='c', channel_mode=ChannelMode.COMPOSITE, default_lut={'cmap': Colormap('viridis')})

# %%
viewer.show()

# %%
# 3d
#viewer.display_model.visible_axes = (X,Y,X)
# 2d
#viewer.display_model.visible_axes = (Y,X)

# %%
#viewer.display_model.channel_mode = "grayscale"

# %%
#viewer.display_model.current_index.update({Y: 0})
