import tifffile
import zarr
import dask.array


FNAME = "big_thing.ome.tiff"
try:
    if data:  # type: ignore
        print("aready loaded data")
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


def metadata():
    def sp(val): return f"{val:_}" if type(val) == type(
        1) or type(val) == type(1.1) else val

    def spmap(val): return tuple(
        map(
            lambda x: sp(x), val)
    )\
        if type(val) == type((1, 1))\
        else list(
        map(
            lambda x: sp(x), val)
    )\
        if type(val) == type([1, 1])\
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
                metadata['series'][i]['dtype'] = str(tif.series[i].dtype)
                metadata['series'][i]['is_multifile'] = tif.series[i].is_multifile
                metadata['series'][i]['is_pyramidal'] = tif.series[i].is_pyramidal
                metadata['series'][i]['kind'] = tif.series[i].kind
                metadata['series'][i]['name'] = tif.series[i].name
                metadata['series'][i]['shape'] = spmap(tif.series[i].shape)
                if hasattr(tif.series[i], '_shape_expanded'):
                    metadata['series'][i]['shape_expanded'] = spmap(
                        tif.series[i]._shape_expanded)  # type: ignore
                metadata['series'][i]['size'] = sp(tif.series[i].size)

        MAX_PAGES = 10
        metadata['pages'] = list()
        for i, _ in enumerate(tif.pages):
            if i > MAX_PAGES:
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
                    if type(val) == type(1):
                        metadata['pages'][i]['tags'][key]['value'] = val
                    elif type(val) == type(1.0):
                        metadata['pages'][i]['tags'][key]['value'] = val
                    else:
                        metadata['pages'][i]['tags'][key]['value'] = str(val)
                else:
                    metadata['pages'][i]['tags'][key]['value'] = "LOTS OF STUFF"
    del tif

    return metadata
