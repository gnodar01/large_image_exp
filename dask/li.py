import os
import pprint
import sys
import termios
import tty
from importlib import reload
from math import ceil

import matplotlib.pyplot as plt
import ometiff_metadata
import tifffile
import zarr
from numpy.typing import NDArray

import dask.array
from dask.array.core import Array as daskArray

FNAME = os.path.abspath('.')
if FNAME.endswith('dask'):
    FNAME = FNAME + "/big_thing.ome.tiff"
else:
    FNAME = FNAME + "/../big_thing.ome.tiff"
__cached_meta = None
try:
    if data:  # type: ignore
        print("aready loaded data")
        reload(ometiff_metadata)
    else:
        print("failed to load data")
except NameError:
    print("loading data")
    _store = tifffile.imread(FNAME, aszarr=True)
    _cache = zarr.LRUStoreCache(_store, max_size=2**29)
    _zobj = zarr.open(_cache, mode="r")
    _zarr_data: list[zarr.Array] = [ # type: ignore
        _zobj[int(dataset["path"])]
        for dataset in _zobj.attrs["multiscales"][0]["datasets"]
    ]
    data: list[daskArray] = [dask.array.from_zarr(z) for z in _zarr_data] # type: ignore
    print("done loading data")


def full_meta(**kwargs):
    return ometiff_metadata.extract_metadata(FNAME, **kwargs)


def meta():
    global __cached_meta
    if __cached_meta is None:
        __cached_meta = ometiff_metadata.extract_standard_metadata(FNAME)
    return __cached_meta


def res():
    global __cached_meta
    if __cached_meta is None:
        _meta = meta()
    else:
        _meta = __cached_meta
    return _meta["resolutions"]

# def level_width(lvl: int):
#    return res()[lvl]["width"]

# def level_height(lvl: int):
#    return res()[lvl]["height"]


def tile_width(lvl: int):
    return res()[lvl]["max_tile_width"]


def tile_height(lvl: int):
    return res()[lvl]["max_tile_height"]


def nx(lvl: int):
    '''num tiles in x direction'''
    return res()[lvl]["n_tiles_x"]


def ix(lvl: int, n: int):
    '''idx of tile in x direction'''
    _tile_width = tile_width(lvl)
    _img_width = res()[lvl]["width"]

    n_tile_cols = ceil(_img_width/_tile_width)

    return n % n_tile_cols


def ny(lvl: int):
    '''num tiles in y direction'''
    return res()[lvl]["n_tiles_y"]


def iy(lvl: int, n: int):
    '''idx of tile in y direction'''
    _tile_width = tile_width(lvl)
    _img_width = res()[lvl]["width"]

    n_tile_cols = ceil(_img_width/_tile_width)

    return n // n_tile_cols


def nn(lvl: int):
    '''num of nth values'''
    return nx(lvl) * ny(lvl)


def n_slices(n, level=0) -> tuple[slice, slice]:
    '''0-indexed, assumes row major'''
    _res = res()
    n_tiles_x = _res[level]["n_tiles_x"]
    tile_row = int(n // n_tiles_x)
    tile_col = int(n % n_tiles_x)

    assert n_tiles_x > 0

    col_start = int(tile_col * _res[level]["max_tile_width"])
    col_end = int(col_start + _res[level]["max_tile_width"])

    row_start = int(tile_row * _res[level]["max_tile_height"])
    row_end = int(row_start + _res[level]["max_tile_height"])

    assert col_end > col_start
    assert row_end > row_start

    return (slice(col_start, col_end, 1), slice(row_start, row_end, 1))


def set_frame(start: int, stop: int | None = None, step: int | None = None, level: int | None = None) -> slice:
    if level:
        max_frame = res()[level]["channels"] - 1
    else:
        max_frame = meta()["c_size"] - 1

    # start can't surpass max_frame, can't go below 0
    start = max(0, min(start, max_frame))

    if stop is None:
        stop = start + 1
    else:
        # end must be at least one greater than start
        stop = max(start + 1, stop)

    if step is None:
        step = 1
    # step is allowed to exceed max_frame, as long as stop is set properly

    return slice(start, stop, step)


def increment_frame(curr_frame: slice, level: int | None) -> slice:
    return set_frame(start = curr_frame.start + 1, stop = curr_frame.stop + 1, step = curr_frame.step, level = level)


def decrement_frame(curr_frame: slice, level: int | None) -> slice:
    return set_frame(start = curr_frame.start - 1, stop = curr_frame.stop - 1, step = curr_frame.step, level = level)


def tile_n(nth: int, frame: slice = slice(0,1,1), level: int = 0, do_transpose=True) -> daskArray:
    assert len(data) > level
    assert level >= 0
    _res = res()
    col_slice, row_slice = n_slices(nth, level)
    row_idx = _res[level]["dims"].index("height")
    col_idx = _res[level]["dims"].index("width")
    frame_idx = _res[level]["dims"].index("sequence")

    standard_idxs = (0, 1, 2)
    assert row_idx != col_idx
    assert row_idx != frame_idx
    assert col_idx != frame_idx
    assert row_idx in standard_idxs
    assert col_idx in standard_idxs
    assert frame_idx in standard_idxs

    idxs: dict[int, slice] = dict()
    idxs[row_idx] = row_slice
    idxs[col_idx] = col_slice
    idxs[frame_idx] = frame

    if do_transpose:
        tile = data[level][idxs[0], idxs[1], idxs[2]].transpose(row_idx, col_idx, frame_idx)
    else:
        tile = data[level][idxs[0], idxs[1], idxs[2]]

    assert 0 not in tile.shape, f"invalid shape {tile.shape}, from idxs {idxs}"

    return tile

## DISPLAY ##


def show(img: daskArray, fig_scale=1.0, min_intensities: NDArray | None=None, max_intensities: NDArray | None=None):
    '''
    width and height in inches
    default WxH: 6.4, 4.8
    '''

    if min_intensities is None:
        _min_intensities: NDArray = img.min(axis=(0,1), keepdims=True).compute()
    else:
        _min_intensities: NDArray = min_intensities
    if max_intensities is None:
        _max_intensities: NDArray = img.max(axis=(0,1), keepdims=True).compute()
    else:
        _max_intensities: NDArray = max_intensities

    _img = ((img - _min_intensities) / (_max_intensities - _min_intensities)).compute()

    DPI = 331
    fig_width = _img.shape[1] * fig_scale / DPI
    fig_height = _img.shape[0] * fig_scale / DPI

    _ = plt.figure(figsize=(fig_width, fig_height), dpi=DPI, frameon=False)  # figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis("off")
    _ = plt.imshow(_img, vmin=0, vmax=1)  # ax
    plt.show()


def get_single_key():
    """Reads a single character from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)  # Read a single character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def viewer(debug=False):
    _meta = meta()
    # tile width/height may be smaller depending on if we're at the edge of an image; these are just the ideal size
    # max_tile_width = _meta["tile_width"]
    # max_tile_height = _meta["tile_height"]

    _res = _meta["resolutions"]

    level = len(_res) - 1
    # level = 0
    frame = set_frame(start=0, level=level)
    nth = 0

    tile = tile_n(nth=nth, frame=frame, level=level)
    if tile is None:
        return

    min_intensities: NDArray = tile.min(axis=(0,1), keepdims=True).compute()
    max_intensities: NDArray = tile.max(axis=(0,1), keepdims=True).compute()

    while True:
        reset_intensities = False

        clear_screen()
        show(
            tile,
            fig_scale=0.5,
            min_intensities=min_intensities,
            max_intensities=max_intensities,
        )

        if debug:
            display_tile = dict(_res[level])
            display_tile["level"] = level
            display_tile["frame"] = frame
            display_tile["nth"] = nth
            display_tile["nlevels"] = len(_res)
            display_tile["nframes"] = _res[level]["channels"]
            display_tile["tile_width"] = tile_width(level)
            display_tile["tile_height"] = tile_height(level)
            display_tile["n_tiles_total"] = nn(level)
            display_tile["global_min"] = min_intensities.flatten()
            display_tile["global_max"] = max_intensities.flatten()
            display_tile["tile_min"] = tile.min(axis=(0,1), keepdims=True).flatten().compute()
            display_tile["tile_max"] = tile.max(axis=(0,1), keepdims=True).flatten().compute()
            display_tile["tile_shape"] = tile.shape
            pprint.pp(display_tile)

        key = get_single_key()
        key_ord = ord(key)

        # quit
        if key == "q":
            clear_screen()
            break

        # next frame
        elif key == "n" or key == "L":
            frame = increment_frame(curr_frame=frame, level=level)
            reset_intensities = True

        # previous frame
        elif key == "p" or key == "H":
            frame = decrement_frame(curr_frame=frame, level=level)
            reset_intensities = True

        # next tile (no bounds check) - cr or nl or sp
        elif key_ord == 13 or key_ord == 10 or key_ord == 32:
            nth = min(nn(level) - 1, nth + 1)

        # tile right (unless edge)
        elif key == "l":
            curr_x = nth % nx(level)
            if curr_x < (nx(level) - 1):
                nth += 1

        # tile left (unless edge)
        elif key == "h":
            curr_x = nth % nx(level)
            if curr_x > 0:
                nth -= 1

        # tile up (unless edge)
        elif key == "k":
            new_nth = nth - nx(level)
            if new_nth >= 0:
                nth = new_nth

        # tile down (unless edge)
        elif key == "j":
            new_nth = nth + nx(level)
            if new_nth < nn(level):
                nth = new_nth

        #  down the inverted pyramid (downscale)
        elif key == "u" or key == "K":
            if level < (len(_res) - 1):
                new_ix = ix(level, nth) // 2
                new_iy = iy(level, nth) // 2

                level += 1

                new_nx = nx(level)

                nth = new_iy * new_nx + new_ix

        # up the inverted pyramid (upscale)
        elif key == "d" or key == "J":
            if level > 0:
                new_ix = ix(level, nth) * 2
                new_iy = iy(level, nth) * 2

                level = max(0, level - 1)

                new_nx = nx(level)

                nth = new_iy * new_nx + new_ix

        # composite of first 3 channels
        elif key == "c":
            frame = set_frame(start=0, stop=3, step=1, level=level)
            reset_intensities = True

        # reset back to single frame on 0
        elif key == "C":
            frame = set_frame(start=0, stop=1, step=1, level=level)
            reset_intensities = True

        # ? - quit
        else:
            clear_screen()
            print("quit unexpectedly", "key:", key, "ord", key_ord)
            break

        assert nth >= 0
        assert nth <= nn(level), f"only {nn(level)} tiles at level {level}, got {nth}"

        tile = tile_n(nth=nth, frame=frame, level=level)

        assert tile is not None

        if reset_intensities:
            min_intensities: NDArray = tile.min(axis=(0,1), keepdims=True).compute()
            max_intensities: NDArray = tile.max(axis=(0,1), keepdims=True).compute()

if __name__ == "__main__":
    print(full_meta(max_pages=None, include_tags=True))
