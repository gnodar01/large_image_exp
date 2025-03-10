import tifffile
import zarr
import dask.array


try:
    if data:  # type: ignore
        print("aready loaded data")
except NameError:
    print("loading data")
    _store = tifffile.imread("big_thing.ome.tiff", aszarr=True)
    _cache = zarr.LRUStoreCache(_store, max_size=2**29)
    _zobj = zarr.open(_cache, mode="r")
    _zarr_data = [
        _zobj[int(dataset["path"])]
        for dataset in _zobj.attrs["multiscales"][0]["datasets"]
    ]
    data = [dask.array.from_zarr(z) for z in _zarr_data]  # type: ignore
