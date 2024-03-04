Note: this is a proof-of-concept, and many things are incomplete, untested, or don't work.

# cubed-xarray

Interface for using [cubed](https://github.com/cubed-dev/cubed) with [xarray](https://github.com/pydata/xarray).

## Requirements

- Cubed version >=0.6.3
- Xarray version >=2024.02.0

## Installation

Install via pip 

`pip install cubed-xarray`

or conda

`conda install -c conda-forge cubed-xarray`

## Importing

You don't need to import this package in user code. Once poperly installed, xarray should automatically become aware of this package via the magic of entrypoints.

## Usage

Xarray objects backed by cubed arrays can be created either by:

1. Passing existing `cubed.Array` objects to the `data` argument of xarray constructors,
2. Calling `.chunk` on xarray objects,
3. Passing a `chunks` argument to `xarray.open_dataset`.

In (2) and (3) the choice to use `cubed.Array` instead of `dask.array.Array` is made by passing the keyword argument `chunked_array_type='cubed'`.
To pass arguments to the constructor of `cubed.Array` you should pass them via the dictionary `from_array_kwargs`, e.g. `from_array_kwargs={'spec': cubed.Spec(allowed_mem='2GB')}`.

If cubed and cubed-xarray are installed but dask is not, then specifying `chunked_array_type` is not necessary, 
as the entrypoints system will then default to the only chunked parallel backend available (i.e. cubed).

## Sharp Edges 🔪

Some things almost certainly won't work yet:
- Certain operations called in xarray but not implemented in cubed, for instance `pad` (see https://github.com/tomwhite/cubed/issues/193)
- Array operations involving NaNs - for now use `skipna=True` to avoid eager loading (see https://github.com/pydata/xarray/issues/7243)
- Using `parallel=True` with `xr.open_mfdataset` won't work because cubed doesn't implement a version of `dask.Delayed` (see https://github.com/pydata/xarray/issues/7810)
- Groupby (see https://github.com/tomwhite/cubed/issues/223 and https://github.com/xarray-contrib/flox/issues/224)
- `xarray.map_blocks` does not actually dispatch to `cubed.map_blocks` yet, and will always use Dask.
- Certain operations using `cumreduction` (e.g. `ffill` and `bfill`) are [not hooked up to the `ChunkManager` yet](https://github.com/tomwhite/cubed/issues/277#issuecomment-1648567431), so will attempt to call dask.

and some other things _might_ work but have not yet been tried:

- Saving to formats other than zarr

In general a bug could take the form of an error, or of a silent attempt to coerce the array type to numpy by immediately computing the underlying array.

## Tests

Integration tests for wrapping cubed with xarray also live in this repository.
