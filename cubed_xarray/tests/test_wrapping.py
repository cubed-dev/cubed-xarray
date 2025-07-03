import sys

import cubed
import pytest
import xarray as xr
from cubed.runtime.create import create_executor
from numpy.testing import assert_array_equal
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


def assert_identical(a, b):
    """A version of this function which accepts numpy arrays"""
    __tracebackhide__ = True
    from xarray.testing import assert_identical as assert_identical_

    if hasattr(a, "identical"):
        assert_identical_(a, b)
    else:
        assert_array_equal(a, b)


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


def identity(x):
    return x


# based on test_apply_dask_parallelized_one_arg
def test_apply_ufunc_parallelized_one_arg():
    array = cubed.ones((2, 2), chunks=(1, 1))
    data_array = xr.DataArray(array, dims=("x", "y"))

    def parallel_identity(x):
        return xr.apply_ufunc(
            identity,
            x,
            output_dtypes=[x.dtype],
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": False},
        )

    actual = parallel_identity(data_array)
    assert isinstance(actual.data, cubed.Array)
    assert actual.data.chunks == array.chunks
    assert_identical(data_array, actual)

    computed = data_array.compute()
    actual = parallel_identity(computed)
    assert_identical(computed, actual)
