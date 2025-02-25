import matplotlib.pyplot as plt
import numpy as np
import os
import termios
import sys
import large_image
import tty # linux/macOS only
from time import sleep
from pathlib import Path

try:
    if src:
        print('aready loaded src')
except NameError:
    print('loading src')
    r = Path('/Users/ngogober/Developer/CellProfiler/large_images')
    fn = 'E02b_Tonsil_Full_P94_A31_C141_LJI_Orion12_ASCM@20230606_220006_885831.ome.tiff'
    p = r / fn

    src = large_image.open(p)


def region(left=0, top=0, width=1024, height=1024, maxWidth=1000, mime=False):
    global src
    if maxWidth:
        arr, mime_type = src.getRegion(
                region=dict(left=left,top=top,width=width,height=height),
                output=dict(maxWidth=maxWidth),
                format=large_image.constants.TILE_FORMAT_NUMPY)
    else:
        arr, mime_type = src.getRegion(
                region=dict(left=left,top=top,width=width,height=height),
                format=large_image.constants.TILE_FORMAT_NUMPY)

    if mime:
        return arr, mime_type
    return arr

# z is the resolution level
def tile(x,y,z):
    meta = src.getMetadata()
    levels = meta.get('levels', None)
    if not levels:
        print('bad thing')
        return
    if z >= levels:
        print('no such level')
        return
    return src.getTile(x,y,z, numpyAllowed='always', sparseFallback=False)

def res():
    meta = src.getMetadata()
    levels = meta.get('levels', None)
    sizeX = meta.get('sizeX', None)
    sizeY = meta.get('sizeY', None)
    tileWidth = meta.get('tileWidth', None)
    tileHeight = meta.get('tileHeight', None)
    if not levels or not sizeX or not sizeY or not tileWidth or not tileHeight:
        print('bad thing')
        return

    resolutions = dict()
    for i in range(levels):
        scale_level = levels-i-1
        scale_factor = 1/(2**scale_level)
        resolutions[i] = dict(width=int(sizeX*scale_factor), height=int(sizeY*scale_factor))

    return resolutions

## DISPLAY ##

# width and height in inches
# default WxH: 6.4, 4.8
def show(img, width=6, height=6):
    figure = plt.figure(figsize=(width,height))
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
    os.system('cls' if os.name == 'nt' else 'clear')

def showi():
    itr = src.tileIterator( format=large_image.constants.TILE_FORMAT_NUMPY )

    first_tile = next(itr)

    do_break = False
    for tile in itr:
        clear_screen()
        show(tile['tile'], width=12, height=12)
        key = get_single_key()

        if key == 'q':
            clear_screen()
            break

