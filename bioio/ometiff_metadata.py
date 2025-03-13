import xmltodict
import tifffile
from math import ceil
from typing import TypedDict, Literal


class Resolution(TypedDict):
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    height: int
    width: int
    channels: int
    max_tile_height: int
    max_tile_width: int
    n_tiles_x: int
    n_tiles_y: int


class StandardMetadata(TypedDict):
    endiness: Literal["<", ">"]
    dim_order: str
    x_idx: int
    y_idx: int
    c_idx: int
    z_idx: int
    t_idx: int
    x_size: int
    y_size: int
    z_size: int
    c_size: int
    t_size: int
    x_mag: float
    x_mag_unit: str
    y_mag: float
    y_mag_unit: str
    z_mag: float
    z_mag_unit: str
    shape: tuple[int, ...]
    dtype: str
    channel_names: tuple[str, ...]
    tile_height: int
    tile_width: int
    resolutions: dict[int, Resolution]


def extract_metadata(FNAME: str, max_pages: int | None = 0, include_tags=False):
    def sp(val): return f"{val:_}" if type(val) is type(
        1) or type(val) is type(1.1) else val

    def spmap(val): return tuple(
        map(lambda x: sp(x), val)
    )\
        if type(val) is type((1, 1))\
        else list(
        map(lambda x: sp(x), val)
    )\
        if type(val) is type([1, 1])\
        else val

    metadata = dict()

    with tifffile.TiffFile(FNAME) as tif:
        metadata['byteorder'] = tif.byteorder
        metadata['num_pages'] = len(tif.pages)
        metadata['num_series'] = len(tif.series)

        is_thing_list = [x for x in dir(tif) if x.startswith('is_')]
        has_metadata_list = [x for x in dir(tif) if x.endswith('_metadata')]
        metadata['kinds'] = list()
        metadata['metadatas'] = dict()
        for is_thing in is_thing_list:
            is_thing_val = getattr(tif, is_thing)
            if is_thing_val:
                metadata['kinds'].append(is_thing[3:])
        for md in has_metadata_list:
            md_val = getattr(tif, md)
            if md_val:
                if md == "ome_metadata":
                    metadata['metadatas'][md[:-9]] = xmltodict.parse(md_val)
                else:
                    metadata['metadatas'][md[:-9]] = md_val

        if metadata['num_series'] > 0:
            metadata['series'] = list()
            for i, _ in enumerate(tif.series):
                metadata['series'].append(dict())
                metadata['series'][i]['axes'] = tif.series[i].axes
                if hasattr(tif.series[i], '_axes_expanded'):
                    metadata['series'][i]['axes_expanded'] = tif.series[i]._axes_expanded  # type: ignore
                if hasattr(tif.series[i], 'axes_expanded'):
                    metadata['series'][i]['axes_expanded'] = tif.series[i]._axes_expanded  # type: ignore
                if hasattr(tif.series[i], 'dims'):
                    metadata['series'][i]['dims'] = tif.series[i].dims
                metadata['series'][i]['dtype'] = str(tif.series[i].dtype)
                metadata['series'][i]['is_multifile'] = tif.series[i].is_multifile
                metadata['series'][i]['kind'] = tif.series[i].kind
                metadata['series'][i]['name'] = tif.series[i].name
                metadata['series'][i]['shape'] = spmap(tif.series[i].shape)
                if hasattr(tif.series[i], '_shape_expanded'):
                    metadata['series'][i]['shape_expanded'] = spmap(
                        tif.series[i]._shape_expanded)  # type: ignore
                metadata['series'][i]['size'] = sp(tif.series[i].size)
                metadata['series'][i]['sizes'] = tif.series[i].sizes
                metadata['series'][i]['is_pyramidal'] = tif.series[i].is_pyramidal
                if metadata['series'][i]['is_pyramidal']:
                    metadata['series'][i]['pyramid'] = dict()
                    metadata['series'][i]['pyramid']['num_levels'] = len(tif.series[i].levels)
                    metadata['series'][i]['pyramid']['levels'] = list()
                    for ii, level in enumerate(tif.series[i].levels):
                        metadata['series'][i]['pyramid']['levels'].append(dict())
                        metadata['series'][i]['pyramid']['levels'][ii]['dims'] = spmap(level.dims)
                        metadata['series'][i]['pyramid']['levels'][ii]['shape'] = spmap(level.shape)
                        metadata['series'][i]['pyramid']['levels'][ii]['size'] = sp(level.size)
                        metadata['series'][i]['pyramid']['levels'][ii]['sizes'] = level.sizes

        metadata['pages'] = list()
        for i, _ in enumerate(tif.pages):
            if max_pages is not None and i >= max_pages:
                break
            metadata['pages'].append(dict())
            metadata['pages'][i]['axes'] = hasattr(
                tif.pages[i], 'axes') and tif.pages[i].axes or 'undefined'
            metadata['pages'][i]['chunked'] = hasattr(
                tif.pages[i], 'chunked') and spmap(tif.pages[i].chunked) or 'undefined'
            metadata['pages'][i]['chunks'] = hasattr(
                tif.pages[i], 'chunks') and spmap(tif.pages[i].chunks) or 'undefined'
            metadata['pages'][i]['compression'] = hasattr(
                tif.pages[i], 'compression') and str(tif.pages[i].compression) or 'undefined'
            if hasattr(tif.pages[i], 'colormap'):
                metadata['pages'][i]['colormap'] = dict()
                if hasattr(tif.pages[i].colormap, 'ndim'):  # type: ignore
                    metadata['pages'][i]['colormap']['ndim'] = tif.pages[i].colormap.ndim  # type: ignore
                else:
                    metadata['pages'][i]['colormap']['ndim'] = 'undefined'
                if hasattr(tif.pages[i].colormap, 'nbytes'):  # type: ignore
                    metadata['pages'][i]['colormap']['nbytes'] = tif.pages[i].colormap.nbytes  # type: ignore
                else:
                    metadata['pages'][i]['colormap']['nbytes'] = 'undefined'
                if hasattr(tif.pages[i].colormap, 'shape'):  # type: ignore
                    metadata['pages'][i]['shape'] = spmap(
                        tif.pages[i].colormap.shape)  # type: ignore
                else:
                    metadata['pages'][i]['shape'] = 'undefined'
            metadata['pages'][i]['dtype'] = hasattr(
                tif.pages[i], 'dtype') and str(tif.pages[i].dtype) or 'undefined'
            metadata['pages'][i]['imagedepth'] = hasattr(
                tif.pages[i], 'imagedepth') and tif.pages[i].imagedepth or 'undefined'  # type: ignore
            metadata['pages'][i]['imagelength'] = hasattr(
                tif.pages[i], 'imagelength') and sp(tif.pages[i].imagelength) or 'undefined'  # type: ignore
            metadata['pages'][i]['imagewidth'] = hasattr(
                tif.pages[i], 'imagewidth') and sp(tif.pages[i].imagewidth) or 'undefined'  # type: ignore
            metadata['pages'][i]['nbytes'] = hasattr(
                tif.pages[i], 'nbytes') and sp(tif.pages[i].nbytes) or 'undefined'
            metadata['pages'][i]['shape'] = hasattr(
                tif.pages[i], 'shape') and spmap(tif.pages[i].shape) or 'undefined'
            metadata['pages'][i]['size'] = hasattr(tif.pages[i].size, "size") and sp(tif.pages[i].size) or "n/a"
            if hasattr(tif.pages[i], 'resolution'):
                metadata['pages'][i]['resolution'] = spmap(tif.pages[i].resolution)  # type: ignore
            if hasattr(tif.pages[i], 'resolutionunit'):
                metadata['pages'][i]['resolutionunit_name'] = tif.pages[i].resolutionunit.name  # type: ignore
                metadata['pages'][i]['resolutionunit_value'] = sp(tif.pages[i].resolutionunit.value)  # type: ignore
            if hasattr(tif.pages[i], 'software'):
                metadata['pages'][i]['software'] = tif.pages[i].software  # type: ignore
            if hasattr(tif.pages[i], 'software'):
                metadata['pages'][i]['tile'] = spmap(tif.pages[i].tile)  # type: ignore
            if hasattr(tif.pages[i], 'tile'):
                metadata['pages'][i]['tilewidth'] = sp(tif.pages[i].tilewidth)  # type: ignore
            if hasattr(tif.pages[i], 'tilewidth'):
                metadata['pages'][i]['tilelength'] = sp(tif.pages[i].tilelength)  # type: ignore

            if include_tags:
                tag_keys = hasattr(
                    tif.pages[i], 'tags') and tif.pages[i].tags.keys() or []  # type: ignore
                metadata['pages'][i]['tags'] = dict()
                for key in tag_keys:
                    metadata['pages'][i]['tags'][key] = dict()  # type: ignore
                    metadata['pages'][i]['tags'][key]['code'] = tif.pages[i].tags[key].code  # type: ignore
                    # type: ignore
                    metadata['pages'][i]['tags'][key]['count'] = tif.pages[i].tags[key].count  # type: ignore
                    metadata['pages'][i]['tags'][key]['dtype_name'] = tif.pages[i].tags[key].dtype_name  # type: ignore
                    metadata['pages'][i]['tags'][key]['name'] = tif.pages[i].tags[key].name  # type: ignore
                    metadata['pages'][i]['tags'][key]['dataformat'] = tif.pages[i].tags[key].dataformat  # type: ignore
                    metadata['pages'][i]['tags'][key]['valuebytecount'] = tif.pages[i].tags[key].valuebytecount  # type: ignore
                    val = tif.pages[i].tags[key].value  # type: ignore
                    if tif.pages[i].tags[key].valuebytecount < 100:  # type: ignore
                        if type(val) is type(1):
                            metadata['pages'][i]['tags'][key]['value'] = val
                        elif type(val) is type(1.0):
                            metadata['pages'][i]['tags'][key]['value'] = val
                        else:
                            metadata['pages'][i]['tags'][key]['value'] = str(val)
                    else:
                        metadata['pages'][i]['tags'][key]['value'] = "LOTS OF STUFF"
    del tif

    return metadata


def extract_standard_metadata(FNAME: str) -> StandardMetadata:
    full_meta = extract_metadata(FNAME, max_pages=None, include_tags=False)
    pixels_meta = full_meta["metadatas"]["ome"]["OME"]["Image"]["Pixels"]

    endiness = "<" if pixels_meta["@BigEndian"] == "false" else ">"
    dim_order = str(pixels_meta["@DimensionOrder"])

    x_idx = dim_order.index("X")
    y_idx = dim_order.index("Y")
    c_idx = dim_order.index("C")
    z_idx = dim_order.index("Z")
    t_idx = dim_order.index("T")

    x_size = int(pixels_meta["@SizeX"])
    y_size = int(pixels_meta["@SizeY"])
    z_size = int(pixels_meta["@SizeZ"])
    c_size = int(pixels_meta["@SizeC"])
    t_size = int(pixels_meta["@SizeT"])

    x_mag = float(pixels_meta["@PhysicalSizeX"])
    x_mag_unit = str(pixels_meta["@PhysicalSizeXUnit"])
    y_mag = float(pixels_meta["@PhysicalSizeY"])
    y_mag_unit = str(pixels_meta["@PhysicalSizeYUnit"])
    z_mag = float(pixels_meta["@PhysicalSizeZ"])
    z_mag_unit = str(pixels_meta["@PhysicalSizeZUnit"])

    shape = [1] * 5
    shape[x_idx] = x_size
    shape[y_idx] = y_size
    shape[c_idx] = c_size
    shape[z_idx] = z_size
    shape[t_idx] = t_size
    shape = tuple(shape)

    dtype = str(pixels_meta["@Type"])

    channel_names = tuple(map(lambda channel_info: str(channel_info["@Name"]), pixels_meta["Channel"]))

    tile_height = int(full_meta["pages"][0]["tilelength"])
    tile_width = int(full_meta["pages"][0]["tilewidth"])

    resolutions: dict[int, Resolution] = dict()

    levels_dim_order = tuple(map(lambda d: str(d), full_meta["series"][0]["dims"]))
    levels_height_idx = levels_dim_order.index("height")
    levels_width_idx = levels_dim_order.index("width")
    levels_seq_idx = levels_dim_order.index("sequence")

    levels = full_meta["series"][0]["pyramid"]["levels"]
    for i, level in enumerate(levels):
        level_shape = tuple(map(lambda v: int(v), level["shape"]))

        level_dims = levels_dim_order
        level_height = level_shape[levels_height_idx]
        level_width = level_shape[levels_width_idx]
        level_channels = level_shape[levels_seq_idx]
        level_max_tile_height = min(tile_height, level_height)
        level_max_tile_width = min(tile_width, level_width)
        level_n_tiles_x = ceil(level_width / level_max_tile_width)
        level_n_tiles_y = ceil(level_height / level_max_tile_height)

        resolutions[i] = {
            "shape": level_shape,
            "dims": level_dims,
            "height": level_height,
            "width": level_width,
            "channels": level_channels,
            "max_tile_height": level_max_tile_height,
            "max_tile_width": level_max_tile_width,
            "n_tiles_x": level_n_tiles_x,
            "n_tiles_y": level_n_tiles_y
        }

    return {
        "endiness": endiness,
        "dim_order": dim_order,
        "x_idx": x_idx,
        "y_idx": y_idx,
        "c_idx": c_idx,
        "z_idx": z_idx,
        "t_idx": t_idx,
        "x_size": x_size,
        "y_size": y_size,
        "z_size": z_size,
        "c_size": c_size,
        "t_size": t_size,
        "x_mag": x_mag,
        "x_mag_unit": x_mag_unit,
        "y_mag": y_mag,
        "y_mag_unit": y_mag_unit,
        "z_mag": z_mag,
        "z_mag_unit": z_mag_unit,
        "shape": shape,
        "dtype": dtype,
        "channel_names": channel_names,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "resolutions": resolutions,
    }
