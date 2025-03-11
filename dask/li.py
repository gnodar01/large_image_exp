import tifffile
import zarr
import dask.array
from importlib import reload
import ometiff_metadata


FNAME = "big_thing.ome.tiff"
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


def meta(full=False, **kwargs):
    if full:
        return ometiff_metadata.extract_metadata(FNAME, **kwargs)
    else:
        return ometiff_metadata.extract_standard_metadata(FNAME)


def res():
    return ometiff_metadata.extract_resolutions(FNAME)
