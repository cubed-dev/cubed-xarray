import sys

import cubed
import pytest
import xarray as xr
from cubed.runtime.create import create_executor
from xarray.namedarray.parallelcompat import list_chunkmanagers
from xarray.tests import assert_allclose, create_test_data

from cubed_xarray.cubedmanager import CubedManager

EXECUTORS = [create_executor("single-threaded")]

if sys.version_info >= (3, 11):
    EXECUTORS.append(create_executor("processes"))


@pytest.fixture(
    scope="module",
    params=EXECUTORS,
    ids=[executor.name for executor in EXECUTORS],
)
def executor(request):
    return request.param


class TestDiscoverCubedManager:
    def test_list_cubedmanager(self):
        chunkmanagers = list_chunkmanagers()
        assert "cubed" in chunkmanagers
        assert isinstance(chunkmanagers["cubed"], CubedManager)

    def test_chunk(self):
        da = xr.DataArray([1, 2], dims="x")
        chunked = da.chunk(x=1, chunked_array_type="cubed")
        assert isinstance(chunked.data, cubed.Array)
        assert chunked.chunksizes == {"x": (1, 1)}

    # TODO test cubed is default when dask not installed

    # TODO test dask is default over cubed when both installed


def test_to_zarr(tmpdir, executor):
    spec = cubed.Spec(allowed_mem="200MB", executor=executor)

    original = create_test_data().chunk(
        chunked_array_type="cubed", from_array_kwargs={"spec": spec}
    )

    filename = tmpdir / "out.zarr"
    original.to_zarr(filename)

    with xr.open_dataset(
        filename,
        chunks="auto",
        engine="zarr",
        chunked_array_type="cubed",
        from_array_kwargs={"spec": spec},
    ) as restored:
        assert isinstance(restored.var1.data, cubed.Array)
        computed = restored.compute()
        assert_allclose(original, computed)


def test_dataset_accessor_visualize(tmp_path):
    spec = cubed.Spec(allowed_mem="200MB")

    ds = create_test_data().chunk(
        chunked_array_type="cubed", from_array_kwargs={"spec": spec}
    )
    assert not (tmp_path / "cubed.svg").exists()
    ds.cubed.visualize(filename=tmp_path / "cubed")
    assert (tmp_path / "cubed.svg").exists()
