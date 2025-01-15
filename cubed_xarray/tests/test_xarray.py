# Adapted from xarray/tests/test_dask.py

from __future__ import annotations

import operator
from contextlib import suppress
from textwrap import dedent

import cubed
import cubed.random
import dill
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.tests import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    mock,
)

try:
    from cubed.testing import raise_if_computes as raise_if_cubed_computes
except ImportError:
    from contextlib import nullcontext

    raise_if_cubed_computes = nullcontext


@pytest.mark.xfail(reason="needs https://github.com/cubed-dev/cubed/pull/545")
def test_raise_if_cubed_computes():
    data = cubed.from_array(np.random.RandomState(0).randn(4, 6), chunks=(2, 2))
    with pytest.raises(RuntimeError, match=r"'compute' was called"):
        with raise_if_cubed_computes():
            data.compute()


class CubedTestCase:
    def assertLazyAnd(self, expected, actual, test):
        test(actual, expected)

        if isinstance(actual, Dataset):
            for k, v in actual.variables.items():
                if k in actual.xindexes:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, cubed.Array)
        elif isinstance(actual, DataArray):
            assert isinstance(actual.data, cubed.Array)
            for k, v in actual.coords.items():
                if k in actual.xindexes:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, cubed.Array)
        elif isinstance(actual, Variable):
            assert isinstance(actual.data, cubed.Array)
        else:
            assert False


class TestVariable(CubedTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_identical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_allclose)

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.values = np.random.RandomState(0).randn(4, 6)
        self.data = cubed.from_array(self.values, chunks=(2, 2))

        self.eager_var = Variable(("x", "y"), self.values)
        self.lazy_var = Variable(("x", "y"), self.data)

    def test_basics(self):
        v = self.lazy_var
        assert self.data is v.data
        assert self.data.chunks == v.chunks
        assert_array_equal(self.values, v)

    def test_copy(self):
        self.assertLazyAndIdentical(self.eager_var, self.lazy_var.copy())
        self.assertLazyAndIdentical(self.eager_var, self.lazy_var.copy(deep=True))

    def test_chunk(self):
        for chunks, expected in [
            ({}, ((2, 2), (2, 2, 2))),
            (3, ((3, 1), (3, 3))),
            ({"x": 3, "y": 3}, ((3, 1), (3, 3))),
            ({"x": 3}, ((3, 1), (2, 2, 2))),
            ({"x": (3, 1)}, ((3, 1), (2, 2, 2))),
        ]:
            rechunked = self.lazy_var.chunk(chunks, chunked_array_type="cubed")
            assert rechunked.chunks == expected
            self.assertLazyAndIdentical(self.eager_var, rechunked)

            expected_chunksizes = {
                dim: chunks for dim, chunks in zip(self.lazy_var.dims, expected)
            }
            assert rechunked.chunksizes == expected_chunksizes

    def test_indexing(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0], v[0])
        self.assertLazyAndIdentical(u[:1], v[:1])
        self.assertLazyAndIdentical(u[[0, 1], [0, 1, 2]], v[[0, 1], [0, 1, 2]])

    def test_squeeze(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u[0].squeeze(), v[0].squeeze())

    def test_equals(self):
        v = self.lazy_var
        assert v.equals(v)
        assert isinstance(v.data, cubed.Array)
        assert v.identical(v)
        assert isinstance(v.data, cubed.Array)

    def test_transpose(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.T, v.T)

    @pytest.mark.xfail(reason="needs pad mode='constant_values' in cubed")
    def test_shift(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.shift(x=2), v.shift(x=2))
        self.assertLazyAndIdentical(u.shift(x=-2), v.shift(x=-2))
        assert v.data.chunks == v.shift(x=1).data.chunks

    def test_roll(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u.roll(x=2), v.roll(x=2))
        # assert v.data.chunks == v.roll(x=1).data.chunks  # TODO: fails

    def test_unary_op(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(-u, -v)
        self.assertLazyAndIdentical(abs(u), abs(v))
        self.assertLazyAndIdentical(u.round(), v.round())

    def test_binary_op(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(2 * u, 2 * v)
        self.assertLazyAndIdentical(u + u, v + v)
        # self.assertLazyAndIdentical(u[0] + u, v[0] + v)  # TODO: fails

    def test_binary_op_bitshift(self) -> None:
        # bit shifts only work on ints so we need to generate
        # new eager and lazy vars
        rng = np.random.default_rng(0)
        values = rng.integers(low=-10000, high=10000, size=(4, 6))
        data = cubed.from_array(values, chunks=(2, 2))
        u = Variable(("x", "y"), values)
        v = Variable(("x", "y"), data)
        self.assertLazyAndIdentical(u << 2, v << 2)
        self.assertLazyAndIdentical(u << 5, v << 5)
        self.assertLazyAndIdentical(u >> 2, v >> 2)
        self.assertLazyAndIdentical(u >> 5, v >> 5)

    def test_repr(self):
        expected = dedent(
            f"""\
            <xarray.Variable (x: 4, y: 6)> Size: 192B
            {self.lazy_var.data!r}"""
        )
        assert expected == repr(self.lazy_var)

    def test_pickle(self):
        # Test that pickling/unpickling does not convert the cubed
        # backend to numpy
        # Use dill since pickle can't handle cubed functions
        a1 = self.lazy_var
        a1.compute()
        assert not a1._in_memory
        a2 = dill.loads(dill.dumps(a1))
        assert_identical(a1, a2)
        assert not a1._in_memory
        assert not a2._in_memory

    def test_reduce(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(u.mean(), v.mean())
        # TODO: other reduce functions need work
        # self.assertLazyAndAllClose(u.std(), v.std())
        # with raise_if_cubed_computes():
        #     actual = v.argmax(dim="x")
        # self.assertLazyAndAllClose(u.argmax(dim="x"), actual)
        # with raise_if_cubed_computes():
        #     actual = v.argmin(dim="x")
        # self.assertLazyAndAllClose(u.argmin(dim="x"), actual)
        # self.assertLazyAndAllClose((u > 1).any(), (v > 1).any())
        # self.assertLazyAndAllClose((u < 1).all("x"), (v < 1).all("x"))
        # with pytest.raises(NotImplementedError, match=r"only works along an axis"):
        #     v.median()
        # with pytest.raises(NotImplementedError, match=r"only works along an axis"):
        #     v.median(v.dims)
        # with raise_if_cubed_computes():
        #     v.reduce(duck_array_ops.mean)

    def test_missing_values(self):
        values = np.array([0, 1, np.nan, 3])
        data = cubed.from_array(values, chunks=(2,))

        eager_var = Variable("x", values)
        lazy_var = Variable("x", data)
        self.assertLazyAndIdentical(eager_var, lazy_var.fillna(lazy_var))
        self.assertLazyAndIdentical(Variable("x", range(4)), lazy_var.fillna(2))
        # self.assertLazyAndIdentical(eager_var.count(), lazy_var.count())  # TODO: doesn't use array API

    def test_concat(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndIdentical(u, Variable.concat([v[:2], v[2:]], "x"))
        self.assertLazyAndIdentical(u[:2], Variable.concat([v[0], v[1]], "x"))
        # TODO: following fail
        # self.assertLazyAndIdentical(u[:2], Variable.concat([u[0], v[1]], "x"))
        # self.assertLazyAndIdentical(u[:2], Variable.concat([v[0], u[1]], "x"))
        # self.assertLazyAndIdentical(
        #     u[:3], Variable.concat([v[[0, 2]], v[[1]]], "x", positions=[[0, 2], [1]])
        # )

    def test_missing_methods(self):
        v = self.lazy_var
        with pytest.raises(AttributeError):
            v.argsort()
        with pytest.raises(AttributeError):
            v[0].item()

    @pytest.mark.xfail(reason="np ufuncs don't delegate to cubed")
    def test_univariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.sin(u), np.sin(v))

    @pytest.mark.xfail(reason="np ufuncs don't delegate to cubed")
    def test_bivariate_ufunc(self):
        u = self.eager_var
        v = self.lazy_var
        self.assertLazyAndAllClose(np.maximum(u, 0), np.maximum(v, 0))
        self.assertLazyAndAllClose(np.maximum(u, 0), np.maximum(0, v))

    @pytest.mark.skip(reason="can't call cubed.compute on anything except cubed arrays")
    def test_compute(self):
        u = self.eager_var
        v = self.lazy_var

        # assert dask.is_dask_collection(v)
        (v2,) = cubed.compute(v + 1)
        # assert not dask.is_dask_collection(v2)

        assert ((u + 1).data == v2.data).all()


class TestDataArrayAndDataset(CubedTestCase):
    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_identical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_allclose)

    def assertLazyAndEqual(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_equal)

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.values = np.random.randn(4, 6)
        self.data = cubed.from_array(self.values, chunks=(2, 2))
        self.eager_array = DataArray(
            self.values, coords={"x": range(4)}, dims=("x", "y"), name="foo"
        )
        self.lazy_array = DataArray(
            self.data, coords={"x": range(4)}, dims=("x", "y"), name="foo"
        )

    def test_chunk(self) -> None:
        for chunks, expected in [
            ({}, ((2, 2), (2, 2, 2))),
            (3, ((3, 1), (3, 3))),
            ({"x": 3, "y": 3}, ((3, 1), (3, 3))),
            ({"x": 3}, ((3, 1), (2, 2, 2))),
            ({"x": (3, 1)}, ((3, 1), (2, 2, 2))),
            ({"x": "16B"}, ((1, 1, 1, 1), (2, 2, 2))),
            ("16B", ((1, 1, 1, 1), (1,) * 6)),
            ("16MB", ((4,), (6,))),
        ]:
            # Test DataArray
            rechunked = self.lazy_array.chunk(chunks, chunked_array_type="cubed")
            assert rechunked.chunks == expected
            self.assertLazyAndIdentical(self.eager_array, rechunked)

            expected_chunksizes = {
                dim: chunks for dim, chunks in zip(self.lazy_array.dims, expected)
            }
            assert rechunked.chunksizes == expected_chunksizes

            # Test Dataset
            lazy_dataset = self.lazy_array.to_dataset()
            eager_dataset = self.eager_array.to_dataset()
            expected_chunksizes = {
                dim: chunks for dim, chunks in zip(lazy_dataset.dims, expected)
            }
            rechunked = lazy_dataset.chunk(chunks, chunked_array_type="cubed")

            # Dataset.chunks has a different return type to DataArray.chunks - see issue #5843
            assert rechunked.chunks == expected_chunksizes
            self.assertLazyAndIdentical(eager_dataset, rechunked)

            assert rechunked.chunksizes == expected_chunksizes

    def test_rechunk(self):
        chunked = self.eager_array.chunk({"x": 2}).chunk(
            {"y": 2}, chunked_array_type="cubed"
        )
        assert chunked.chunks == ((2,) * 2, (2,) * 3)
        self.assertLazyAndIdentical(self.lazy_array, chunked)

    def test_new_chunk(self):
        chunked = self.eager_array.chunk(chunked_array_type="cubed")
        assert chunked.data.name.startswith("array-")

    def test_lazy_dataset(self):
        lazy_ds = Dataset({"foo": (("x", "y"), self.data)})
        assert isinstance(lazy_ds.foo.variable.data, cubed.Array)

    def test_lazy_array(self):
        u = self.eager_array
        v = self.lazy_array

        self.assertLazyAndAllClose(u, v)
        self.assertLazyAndAllClose(-u, -v)
        self.assertLazyAndAllClose(u.T, v.T)
        self.assertLazyAndAllClose(u.mean(), v.mean())
        self.assertLazyAndAllClose(1 + u, 1 + v)

        actual = xr.concat([v[:2], v[2:]], "x")
        self.assertLazyAndAllClose(u, actual)

    @pytest.mark.skip(reason="can't call cubed.compute on anything except cubed arrays")
    def test_compute(self):
        u = self.eager_array
        v = self.lazy_array

        # assert dask.is_dask_collection(v)
        (v2,) = cubed.compute(v + 1)
        # assert not dask.is_dask_collection(v2)

        assert ((u + 1).data == v2.data).all()

    def test_groupby(self):
        u = self.eager_array
        v = self.lazy_array

        expected = u.groupby("x").mean(...)
        with raise_if_cubed_computes():
            actual = v.groupby("x").mean(...)
        self.assertLazyAndAllClose(expected, actual)

    @pytest.mark.xfail(reason="needs pad mode='constant_values' in cubed")
    def test_rolling(self):
        u = self.eager_array
        v = self.lazy_array

        expected = u.rolling(x=2).mean()
        with raise_if_cubed_computes():
            actual = v.rolling(x=2).mean()
        self.assertLazyAndAllClose(expected, actual)

    @pytest.mark.xfail(reason="failure in cubed")
    @pytest.mark.parametrize("func", ["first", "last"])
    def test_groupby_first_last(self, func):
        method = operator.methodcaller(func)
        u = self.eager_array
        v = self.lazy_array

        for coords in [u.coords, v.coords]:
            coords["ab"] = ("x", ["a", "a", "b", "b"])
        expected = method(u.groupby("ab"))

        with raise_if_cubed_computes():
            actual = method(v.groupby("ab"))
        self.assertLazyAndAllClose(expected, actual)

        with raise_if_cubed_computes():
            actual = method(v.groupby("ab"))
        self.assertLazyAndAllClose(expected, actual)

    @pytest.mark.xfail(reason="isn't lazy")
    def test_reindex(self):
        u = self.eager_array.assign_coords(y=range(6))
        v = self.lazy_array.assign_coords(y=range(6))

        for kwargs in [
            {"x": [2, 3, 4]},
            {"x": [1, 100, 2, 101, 3]},
            {"x": [2.5, 3, 3.5], "y": [2, 2.5, 3]},
        ]:
            expected = u.reindex(**kwargs)
            actual = v.reindex(**kwargs)
            self.assertLazyAndAllClose(expected, actual)

    def test_to_dataset_roundtrip(self):
        u = self.eager_array
        v = self.lazy_array

        expected = u.assign_coords(x=u["x"])
        self.assertLazyAndEqual(expected, v.to_dataset("x").to_dataarray("x"))

    def test_merge(self):
        def duplicate_and_merge(array):
            return xr.merge([array, array.rename("bar")]).to_dataarray()

        expected = duplicate_and_merge(self.eager_array)
        actual = duplicate_and_merge(self.lazy_array)
        self.assertLazyAndEqual(expected, actual)

    @pytest.mark.xfail(reason="np ufuncs don't delegate to cubed")
    def test_ufuncs(self):
        u = self.eager_array
        v = self.lazy_array
        self.assertLazyAndAllClose(np.sin(u), np.sin(v))

    @pytest.mark.xfail(reason="failure in cubed")
    def test_where_dispatching(self):
        a = np.arange(10)
        b = a > 3
        x = cubed.from_array(a, 5)
        y = cubed.from_array(b, 5)
        expected = DataArray(a).where(b)
        self.assertLazyAndEqual(expected, DataArray(a).where(y))
        self.assertLazyAndEqual(expected, DataArray(x).where(b))
        self.assertLazyAndEqual(expected, DataArray(x).where(y))

    def test_duplicate_dims(self):
        data = np.random.normal(size=(4, 4))
        arr = DataArray(data, dims=("x", "x"))
        chunked_array = arr.chunk({"x": 2})
        assert chunked_array.chunks == ((2, 2), (2, 2))
        assert chunked_array.chunksizes == {"x": (2, 2)}

    def test_stack(self):
        data = cubed.random.random(size=(2, 3, 4), chunks=(1, 3, 4))
        arr = DataArray(data, dims=("w", "x", "y"))
        stacked = arr.stack(z=("x", "y"))
        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)], names=["x", "y"])
        expected = DataArray(cubed.reshape(data, (2, -1)), {"z": z}, dims=["w", "z"])
        assert stacked.data.chunks == expected.data.chunks
        self.assertLazyAndEqual(expected, stacked)

    @pytest.mark.xfail(reason="relies on np.einsum which is not in cubed")
    def test_dot(self):
        eager = self.eager_array.dot(self.eager_array[0])
        lazy = self.lazy_array.dot(self.lazy_array[0])
        self.assertLazyAndAllClose(eager, lazy)

    def test_dataarray_repr(self):
        data = cubed.asarray([1], chunks=1)
        data.name = "array-0"  # change name to something fixed for the repr
        nonindex_coord = cubed.asarray([1], chunks=1)
        a = DataArray(data, dims=["x"], coords={"y": ("x", nonindex_coord)})
        expected = dedent(
            f"""\
            <xarray.DataArray 'array-0' (x: 1)> Size: 8B
            {data!r}
            Coordinates:
                y        (x) int64 8B cubed.Array<chunksize=(1,)>
            Dimensions without coordinates: x"""
        )
        assert expected == repr(a)

    def test_dataset_repr(self):
        data = cubed.asarray([1], chunks=1)
        nonindex_coord = cubed.asarray([1], chunks=1)
        ds = Dataset(data_vars={"a": ("x", data)}, coords={"y": ("x", nonindex_coord)})
        expected = dedent(
            """\
            <xarray.Dataset> Size: 16B
            Dimensions:  (x: 1)
            Coordinates:
                y        (x) int64 8B cubed.Array<chunksize=(1,)>
            Dimensions without coordinates: x
            Data variables:
                a        (x) int64 8B cubed.Array<chunksize=(1,)>"""
        )
        assert expected == repr(ds)

    def test_dataarray_pickle(self):
        # Test that pickling/unpickling converts the cubed backend
        # to numpy in neither the data variable nor the non-index coords
        data = cubed.asarray([1], chunks=1)
        nonindex_coord = cubed.asarray([1], chunks=1)
        a1 = DataArray(data, dims=["x"], coords={"y": ("x", nonindex_coord)})
        a1.compute()
        assert not a1._in_memory
        assert not a1.coords["y"]._in_memory
        a2 = dill.loads(dill.dumps(a1))
        assert_identical(a1, a2)
        assert not a1._in_memory
        assert not a2._in_memory
        assert not a1.coords["y"]._in_memory
        assert not a2.coords["y"]._in_memory

    def test_dataset_pickle(self):
        # Test that pickling/unpickling converts the cubed backend
        # to numpy in neither the data variables nor the non-index coords
        data = cubed.asarray([1], chunks=1)
        nonindex_coord = cubed.asarray([1], chunks=1)
        ds1 = Dataset(data_vars={"a": ("x", data)}, coords={"y": ("x", nonindex_coord)})
        ds1.compute()
        assert not ds1["a"]._in_memory
        assert not ds1["y"]._in_memory
        ds2 = dill.loads(dill.dumps(ds1))
        assert_identical(ds1, ds2)
        assert not ds1["a"]._in_memory
        assert not ds2["a"]._in_memory
        assert not ds1["y"]._in_memory
        assert not ds2["y"]._in_memory

    def test_dataarray_getattr(self):
        # ipython/jupyter does a long list of getattr() calls to when trying to
        # represent an object.
        # Make sure we're not accidentally computing cubed variables.
        data = cubed.asarray([1], chunks=1)
        nonindex_coord = cubed.asarray([1], chunks=1)
        a = DataArray(data, dims=["x"], coords={"y": ("x", nonindex_coord)})
        with raise_if_cubed_computes():
            with suppress(AttributeError):
                getattr(a, "NOTEXIST")

    def test_dataset_getattr(self):
        # Test that pickling/unpickling converts the cubed backend
        # to numpy in neither the data variables nor the non-index coords
        data = cubed.asarray([1], chunks=1)
        nonindex_coord = cubed.asarray([1], chunks=1)
        ds = Dataset(data_vars={"a": ("x", data)}, coords={"y": ("x", nonindex_coord)})
        with raise_if_cubed_computes():
            with suppress(AttributeError):
                getattr(ds, "NOTEXIST")

    def test_values(self):
        # Test that invoking the values property does not convert the cubed
        # backend to numpy
        a = DataArray([1, 2]).chunk()
        assert not a._in_memory
        assert a.values.tolist() == [1, 2]
        assert not a._in_memory

    def test_from_cubed_variable(self):
        # Test array creation from Variable with cubed backend.
        # This is used e.g. in broadcast()
        a = DataArray(self.lazy_array.variable, coords={"x": range(4)}, name="foo")
        self.assertLazyAndIdentical(self.lazy_array, a)


@pytest.mark.parametrize("method", ["load", "compute"])
def test_cubed_kwargs_variable(method):
    chunked_array = cubed.from_array(np.arange(3), chunks=(2,))
    x = Variable("y", chunked_array)
    # args should be passed on to cubed.compute() (via CubedManager.compute())
    with mock.patch.object(
        cubed, "compute", return_value=(np.arange(3),)
    ) as mock_compute:
        getattr(x, method)(foo="bar")
    mock_compute.assert_called_with(chunked_array, foo="bar")


@pytest.mark.parametrize("method", ["load", "compute"])
def test_cubed_kwargs_dataarray(method):
    data = cubed.from_array(np.arange(3), chunks=(2,))
    x = DataArray(data)
    if method in ["load", "compute"]:
        cubed_func = "cubed.compute"
    # args should be passed on to "cubed_func"
    with mock.patch(cubed_func) as mock_func:
        getattr(x, method)(foo="bar")
    mock_func.assert_called_with(data, foo="bar")


@pytest.mark.parametrize("method", ["load", "compute"])
def test_cubed_kwargs_dataset(method):
    data = cubed.from_array(np.arange(3), chunks=(2,))
    x = Dataset({"x": (("y"), data)})
    if method in ["load", "compute"]:
        cubed_func = "cubed.compute"
    # args should be passed on to "cubed_func"
    with mock.patch(cubed_func) as mock_func:
        getattr(x, method)(foo="bar")
    mock_func.assert_called_with(data, foo="bar")


def test_basic_compute():
    ds = Dataset({"foo": ("x", range(5)), "bar": ("x", range(5))}).chunk({"x": 2})
    ds.compute()
    ds.foo.compute()
    ds.foo.variable.compute()


def make_da():
    da = xr.DataArray(
        np.ones((10, 20)),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(100, 120)},
        name="a",
    ).chunk({"x": 4, "y": 5})
    da.x.attrs["long_name"] = "x"
    da.attrs["test"] = "test"
    da.coords["c2"] = 0.5
    da.coords["ndcoord"] = da.x * 2
    da.coords["cxy"] = (da.x * da.y).chunk({"x": 4, "y": 5})

    return da


def make_ds():
    map_ds = xr.Dataset()
    map_ds["a"] = make_da()
    map_ds["b"] = map_ds.a + 50
    map_ds["c"] = map_ds.x + 20
    map_ds = map_ds.chunk({"x": 4, "y": 5})
    map_ds["d"] = ("z", [1, 1, 1, 1])
    map_ds["z"] = [0, 1, 2, 3]
    map_ds["e"] = map_ds.x + map_ds.y
    map_ds.coords["c1"] = 0.5
    map_ds.coords["cx"] = ("x", np.arange(len(map_ds.x)))
    map_ds.coords["cx"].attrs["test2"] = "test2"
    map_ds.attrs["test"] = "test"
    map_ds.coords["xx"] = map_ds["a"] * map_ds.y

    map_ds.x.attrs["long_name"] = "x"
    map_ds.y.attrs["long_name"] = "y"

    return map_ds


# fixtures cannot be used in parametrize statements
# instead use this workaround
# https://docs.pytest.org/en/latest/deprecations.html#calling-fixtures-directly
@pytest.fixture
def map_da():
    return make_da()


@pytest.fixture
def map_ds():
    return make_ds()


def test_unify_chunks(map_ds):
    ds_copy = map_ds.copy()
    ds_copy["cxy"] = ds_copy.cxy.chunk({"y": 10})

    with pytest.raises(ValueError, match=r"inconsistent chunks"):
        ds_copy.chunks

    expected_chunks = {"x": (4, 4, 2), "y": (5, 5, 5, 5)}
    with raise_if_cubed_computes():
        actual_chunks = ds_copy.unify_chunks().chunks
    assert actual_chunks == expected_chunks
    assert_identical(map_ds, ds_copy.unify_chunks())

    out_a, out_b = xr.unify_chunks(ds_copy.cxy, ds_copy.drop_vars("cxy"))
    assert out_a.chunks == ((4, 4, 2), (5, 5, 5, 5))
    assert out_b.chunks == expected_chunks

    # TODO: following fails
    # # Test unordered dims
    # da = ds_copy["cxy"]
    # out_a, out_b = xr.unify_chunks(da.chunk({"x": -1}), da.T.chunk({"y": -1}))
    # assert out_a.chunks == ((4, 4, 2), (5, 5, 5, 5))
    # assert out_b.chunks == ((5, 5, 5, 5), (4, 4, 2))

    # # Test mismatch
    # with pytest.raises(ValueError, match=r"Dimension 'x' size mismatch: 10 != 2"):
    #     xr.unify_chunks(da, da.isel(x=slice(2)))


@pytest.mark.parametrize("obj", [make_ds(), make_da()])
@pytest.mark.parametrize(
    "transform", [lambda x: x.compute(), lambda x: x.unify_chunks()]
)
def test_unify_chunks_shallow_copy(obj, transform):
    obj = transform(obj)
    unified = obj.unify_chunks()
    assert_identical(obj, unified) and obj is not obj.unify_chunks()


@pytest.mark.parametrize("obj", [make_da()])
def test_auto_chunk_da(obj):
    actual = obj.chunk("auto").data
    expected = obj.data.rechunk("auto")
    np.testing.assert_array_equal(actual, expected)
    assert actual.chunks == expected.chunks
