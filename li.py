import os
from pathlib import Path
import pprint
import sys
import termios
import tty
from typing import Iterator, Literal, overload

import large_image
import large_image.constants
from large_image.tilesource.tileiterator import LazyTileDict
import matplotlib.pyplot as plt

try:
    if src: # type: ignore
        print("aready loaded src")
except NameError:
    print("loading src")
    r = Path("/Users/ngogober/Developer/CellProfiler/large_images")
    fn = (
        "E02b_Tonsil_Full_P94_A31_C141_LJI_Orion12_ASCM@20230606_220006_885831.ome.tiff"
    )
    p = r / fn

    src = large_image.open(p)


def srcMetadata():
    return src.getInternalMetadata()


def standardMetadata():
    return src.getMetadata()


def region(frame=0, left=0, top=0, width=1024, height=1024, maxWidth=1000, mime=False):
    global src
    if maxWidth:
        arr, mime_type = src.getRegion(
            frame=frame,
            region=dict(left=left, top=top, width=width, height=height),
            output=dict(maxWidth=maxWidth),
            format=large_image.constants.TILE_FORMAT_NUMPY,
        )
    else:
        arr, mime_type = src.getRegion(
            region=dict(left=left, top=top, width=width, height=height),
            format=large_image.constants.TILE_FORMAT_NUMPY,
        )

    if mime:
        return arr, mime_type
    return arr


# z is the resolution level
def tile_at(x, y, z):
    meta = standardMetadata()
    levels = meta.get("levels", None)
    if not levels:
        print("bad thing")
        return
    if z >= levels:
        print("no such level")
        return
    return src.getTile(x, y, z, numpyAllowed="always", sparseFallback=False)


def res():
    meta = standardMetadata()
    nlevels = meta.get("levels", None)
    sizeX = meta.get("sizeX", None)
    sizeY = meta.get("sizeY", None)
    tileWidth = meta.get("tileWidth", None)
    tileHeight = meta.get("tileHeight", None)
    native_magnification = meta.get("magnification", None)
    assert (
        nlevels
        and sizeX
        and sizeY
        and tileWidth
        and tileHeight
        and native_magnification
    ), "something went wrong with getting resolutions"

    resolutions: dict[int, dict[str, int | float]] = dict()
    # for i in range(nlevels):
    #    scale_level = nlevels-i-1
    #    # don't truncate here, magnification needs float
    #    scale_factor = 1/(2**scale_level)
    #    resolutions[i] = dict(width=int(sizeX*scale_factor), height=int(sizeY*scale_factor), magnification=native_magnification*scale_factor)
    for level in range(nlevels):
        mag = src.getMagnificationForLevel(level)
        magnification = mag["magnification"]
        assert magnification, "No magnification"
        # DO NOT ACCESS tile['tile'] or tile['format'] - would loazy load actual data
        tile = tile_n(magnification=magnification)

        assert tile, "Couldn't get tile"
        assert (
            mag["scale"]
            and tile["width"]
            and tile["height"]
            and tile["iterator_range"]
            and tile["iterator_range"]["level_x_max"]
            and tile["iterator_range"]["level_y_max"]
        ), "Bad mag or tile data"

        resolutions[level] = dict(
            magnification=magnification,
            width=int(sizeX / mag["scale"]),
            height=int(sizeY / mag["scale"]),
            # might be smaller than tileWidth/tileHeight because iamge at level might be smaller than that
            # but it's the max width/height of a tile at that level, because nth=0
            max_tile_width=tile["width"],
            max_tile_height=tile["height"],
            # number of tiles in [x|y] direction
            n_tiles_x=tile["iterator_range"]["level_x_max"],
            n_tiles_y=tile["iterator_range"]["level_y_max"],
            # ignoring level_[x|y]_min for now
            # ignoring overlap for now
        )

    return resolutions


@overload
def tile_n(
    nth: int = 0,
    frame: int = 0,
    magnification: int | float | None = None,
    *,
    return_iter: Literal[False] = False,
) -> LazyTileDict | None: ...


@overload
def tile_n(
    nth: int = 0,
    frame: int = 0,
    magnification: int | float | None = None,
    *,
    return_iter: Literal[True],
) -> Iterator[LazyTileDict] | None: ...


def tile_n(nth=0, frame=0, magnification=None, *, return_iter=False):
    """
    warning: data is loaded lazily, when tile['tile'] or tile['format'] is accessed
    """
    if magnification is not None:
        scale = dict(magnification=magnification)
    else:
        scale = None
    itr = src.tileIterator(
        tile_position=nth,
        frame=frame,
        scale=scale,
        format=large_image.constants.TILE_FORMAT_NUMPY,
    )
    try:
        tile = itr if return_iter else next(itr)
        return tile
    except:
        print("No tile", "nth", nth, "frame", frame, "mag", magnification)
        return None


## DISPLAY ##


# width and height in inches
# default WxH: 6.4, 4.8
def show(img, width=6, height=6):
    figure = plt.figure(figsize=(width, height))
    ax = plt.imshow(img)
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


def showi():
    itr = src.tileIterator(format=large_image.constants.TILE_FORMAT_NUMPY)

    for tile in itr:
        clear_screen()
        show(tile["tile"], width=12, height=12)
        display_tile = {k: v for k, v in tile.items() if k != "tile"}

        pprint.pp(display_tile)

        key = get_single_key()

        if key == "q":
            clear_screen()
            break


def viewer(debug=False):
    meta = standardMetadata()
    # tile width/height may be smaller depending on if we're at the edge of an image; these are just the ideal size
    max_tile_width = meta.get("tileWidth", 1024)
    max_tile_height = meta.get("tileHeight", 1024)

    nlevels = meta.get("levels", 0)
    nframes = len(meta.get("frames", []))
    resolutions = res()

    # level = nlevels - 1
    level = 0
    frame = 0
    nth = 0

    tile = tile_n(
        nth=nth, frame=frame, magnification=resolutions[level]["magnification"]
    )
    if tile is None:
        return

    level_width = lambda lvl: resolutions[lvl]["width"]
    level_height = lambda lvl: resolutions[lvl]["height"]
    tile_width = lambda t: t["width"]
    tile_height = lambda t: t["height"]
    # num tiles in x direction
    nx = lambda t: t["iterator_range"]["level_x_max"]
    # idx of tile in x direction
    ix = lambda t: t["tile_position"]["level_x"]
    # num tiles in y direction
    ny = lambda t: t["iterator_range"]["level_y_max"]
    # idx of tile in y direction
    iy = lambda t: t["tile_position"]["level_y"]
    # num of nth values
    nn = lambda t: nx(t) * ny(t)

    while True:
        clear_screen()
        show(tile["tile"], width=12, height=12)
        display_tile = {k: v for k, v in tile.items() if k != "tile"}
        display_tile["frame_no"] = frame
        display_tile["frame_max"] = nframes

        if debug:
            pprint.pp(display_tile)
            pprint.pp(
                dict(
                    nlevels=nlevels,
                    nframes=nframes,
                    level=level,
                    frame=frame,
                    nth=nth,
                    tile_width=tile_width(tile),
                    tile_height=tile_height(tile),
                    nx=nx(tile),
                    ny=ny(tile),
                    nn=nn(tile),
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
            nth = min(nn(tile) - 1, nth + 1)

        # tile right (unless edge)
        elif key == "l":
            curr_x = nth % nx(tile)
            if curr_x < (nx(tile) - 1):
                nth += 1

        # tile left (unless edge)
        elif key == "h":
            curr_x = nth % nx(tile)
            if curr_x > 0:
                nth -= 1

        # tile up (unless edge)
        elif key == "k":
            new_nth = nth - nx(tile)
            if new_nth >= 0:
                nth = new_nth

        # tile down (unless edge)
        elif key == "j":
            new_nth = nth + nx(tile)
            if new_nth < nn(tile):
                nth = new_nth

        #  up pyramid (downscale)
        elif key == "u" or key == "K":
            if level > 0:
                level = max(0, level - 1)

                new_ix = ix(tile) // 2
                new_iy = iy(tile) // 2
                new_nx = resolutions[level]["n_tiles_x"]

                nth = new_iy * new_nx + new_ix

        # down pyramid (upscale)
        elif key == "d" or key == "J":
            if level < (nlevels - 1):
                level += 1

                new_ix = ix(tile) * 2
                new_iy = iy(tile) * 2
                new_nx = resolutions[level]["n_tiles_x"]

                nth = new_iy * new_nx + new_ix

        # ? - quit
        else:
            clear_screen()
            print("quit unexpectedly", "key:", key, "ord", key_ord)
            break

        tile = tile_n(
            nth=nth, frame=frame, magnification=resolutions[level]["magnification"]
        )
        if tile is None:
            break
