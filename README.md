Note: this is a proof-of-concept, and many things are incomplete, untested, or don't work.

# cubed-xarray

Interface for using [cubed](https://github.com/tomwhite/cubed) with [xarray](https://github.com/pydata/xarray).

## Requirements

Cubed version >v0.6.0

[Xarray pull request #7019](https://github.com/pydata/xarray/pull/7019)

## Installation

Install via pip.

## Importing

You don't need to import this package in user code. Once `pip install`-ed, xarray should automatically become aware of this package via the magic of entrypoints.

## Usage

Xarray objects backed by cubed arrays can be created either by:

1. Passing existing `cubed.Array` objects to the `data` argument of xarray constructors,
2. Calling `.chunk` on xarray objects,
3. Passing a `chunks` argument to `xarray.open_dataset`.

In (2) and (3) the choice to use `cubed.Array` instead of `dask.array.Array` is made by passing the keyword argument `chunked_array_type='cubed'`.
To pass arguments to the constructor of `cubed.Array` then pass them via the dictionary `from_array_kwargs`, e.g. `from_array_kwargs={'spec': cubed.Spec(max_mem=2_000_000)}`.

If cubed and cubed-xarray are installed but dask is not, then specifying the parallel array type to use is not necessary.

## Tests

Integration tests for wrapping cubed with xarray also live in this repository.
