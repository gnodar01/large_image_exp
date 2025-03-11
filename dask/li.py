import os
import sys
from math import ceil
import tty
import termios
import pprint
import tifffile
import zarr
import dask.array
from importlib import reload
import matplotlib.pyplot as plt
import ometiff_metadata


FNAME = "big_thing.ome.tiff"
__cached_meta = None
try:
    if data:  # type: ignore
        print("aready loaded data")
        reload(ometiff_metadata)
except NameError:
    print("loading data")
    _store = tifffile.imread(FNAME, aszarr=True)
    _cache = zarr.LRUStoreCache(_store, max_size=2**29)
    _zobj = zarr.open(_cache, mode="r")
    _zarr_data = [
        _zobj[int(dataset["path"])]
        for dataset in _zobj.attrs["multiscales"][0]["datasets"]
    ]
    data = [dask.array.from_zarr(z) for z in _zarr_data]  # type: ignore


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


# 0-indexed, assumes row major
def n_slices(n, level=0) -> tuple[slice, slice]:
    _res = res()
    n_tiles_x = _res[level]["n_tiles_x"]
    tile_row = int(n // n_tiles_x)
    tile_col = int(n % n_tiles_x)

    col_start = int(tile_col * _res[level]["max_tile_width"])
    col_end = int(col_start + _res[level]["max_tile_width"])

    row_start = int(tile_row * _res[level]["max_tile_height"])
    row_end = int(row_start + _res[level]["max_tile_height"])

    return (slice(col_start, col_end, 1), slice(row_start, row_end, 1))


def tile_n(nth: int, frame: int = 0, level: int = 0) -> zarr.Array:
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

    idxs: dict[int, int | slice] = dict()
    idxs[row_idx] = row_slice
    idxs[col_idx] = col_slice
    idxs[frame_idx] = frame

    return data[level][idxs[0], idxs[1], idxs[2]]

## DISPLAY ##


# width and height in inches
# default WxH: 6.4, 4.8
def show(img, width=6, height=6, min_intensity=None, max_intensity=None):
    # if min_intensity is None:
    #    min_intensity = img.min()
    # if max_intensity is None:
    #    max_intensity = img.max()

    _ = plt.figure(figsize=(width, height))  # figure
    print(
        "local", img.min(), ",", img.max(), "global", min_intensity, ",", max_intensity
    )
    _ = plt.imshow(img, vmin=min_intensity, vmax=max_intensity)  # ax
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
    nframes = _meta["c_size"]

    # level = nlevels - 1
    level = 0
    frame = 0
    nth = 0

    tile = tile_n(nth=nth, frame=frame, level=level)
    if tile is None:
        return

    min_intensity: int = tile.min()  # type: ignore
    max_intensity: int = tile.max()  # type: ignore

    # def level_width(lvl):
    #    return resolutions[lvl]["width"]

    # def level_height(lvl):
    #    return resolutions[lvl]["height"]

    def tile_width(lvl: int):
        return _res[lvl]["width"]

    def tile_height(lvl: int):
        return _res[lvl]["height"]

    # num tiles in x direction
    def nx(lvl: int):
        return _res[lvl]["n_tiles_x"]

    # idx of tile in x direction
    def ix(lvl: int, n: int):
        _tile_width = tile_width(lvl)
        _img_width = _meta["x_size"]

        n_tile_cols = ceil(_img_width/_tile_width)

        return n % n_tile_cols

    # num tiles in y direction
    def ny(lvl: int):
        return _res[lvl]["n_tiles_y"]

    # idx of tile in y direction
    def iy(lvl: int, n: int):
        _tile_height = tile_height(lvl)
        _img_height = _meta["y_size"]

        n_tile_rows = ceil(_img_height/_tile_height)

        return n % n_tile_rows

    # num of nth values
    def nn(lvl: int):
        return nx(lvl) * ny(lvl)

    while True:
        clear_screen()
        show(
            tile,
            width=12,
            height=12,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
        )
        display_tile = dict(_res[level])
        display_tile["frame_no"] = frame
        display_tile["frame_max"] = nframes
        display_tile["level"] = level
        display_tile["frame"] = frame
        display_tile["n"] = nth

        if debug:
            pprint.pp(display_tile)
            pprint.pp(
                dict(
                    nlevels=len(_res),
                    nframes=nframes,
                    level=level,
                    frame=frame,
                    nth=nth,
                    tile_width=tile_width(level),
                    tile_height=tile_height(level),
                    nx=nx(level),
                    ny=ny(level),
                    nn=nn(level),
                )
            )

        key = get_single_key()
        key_ord = ord(key)

        # quit
        if key == "q":
            clear_screen()
            break

        # next frame
        elif key == "n" or key == "L":
            frame = min(nframes - 1, frame + 1)

        # previous frame
        elif key == "p" or key == "H":
            frame = max(0, frame - 1)

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

        #  up pyramid (downscale)
        elif key == "u" or key == "K":
            if level > 0:
                level = max(0, level - 1)

                new_ix = ix(level, nth) // 2
                new_iy = iy(level, nth) // 2
                new_nx = _res[level]["n_tiles_x"]

                nth = new_iy * new_nx + new_ix

        # down pyramid (upscale)
        elif key == "d" or key == "J":
            if level < (len(_res) - 1):
                level += 1

                new_ix = ix(level, nth) * 2
                new_iy = iy(level, nth) * 2
                new_nx = _res[level]["n_tiles_x"]

                nth = new_iy * new_nx + new_ix

        # ? - quit
        else:
            clear_screen()
            print("quit unexpectedly", "key:", key, "ord", key_ord)
            break

        tile = tile_n(nth=nth, frame=frame, level=level)
        if tile is None:
            break
