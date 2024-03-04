import xarray as xr
from xarray.namedarray.parallelcompat import list_chunkmanagers
import cubed

from cubed_xarray.cubedmanager import CubedManager


class TestDiscoverCubedManager:
    def test_list_cubedmanager(self):
        chunkmanagers = list_chunkmanagers()
        assert 'cubed' in chunkmanagers
        assert isinstance(chunkmanagers['cubed'], CubedManager)

    def test_chunk(self):
        da = xr.DataArray([1, 2], dims='x')
        chunked = da.chunk(x=1, chunked_array_type='cubed')
        assert isinstance(chunked.data, cubed.Array)
        assert chunked.chunksizes == {'x': (1, 1)}

    # TODO test cubed is default when dask not installed

    # TODO test dask is default over cubed when both installed
