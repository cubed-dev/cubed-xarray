from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

from tlz import partition

from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint


if TYPE_CHECKING:
    from xarray.core.types import T_Chunks, T_NormalizedChunks
    from cubed import Array as CubedArray


class CubedManager(ChunkManagerEntrypoint["CubedArray"]):
    array_cls: type["CubedArray"]

    def __init__(self) -> None:
        from cubed import Array

        self.array_cls = Array

    def chunks(self, data: "CubedArray") -> T_NormalizedChunks:
        return data.chunks

    def normalize_chunks(
        self,
        chunks: T_Chunks | T_NormalizedChunks,
        shape: tuple[int, ...] | None = None,
        limit: int | None = None,
        dtype: np.dtype | None = None,
        previous_chunks: T_NormalizedChunks | None = None,
    ) -> T_NormalizedChunks:
        from cubed.vendor.dask.array.core import normalize_chunks

        return normalize_chunks(
            chunks,
            shape=shape,
            limit=limit,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )

    def from_array(self, data: np.ndarray, chunks, **kwargs) -> "CubedArray":
        from cubed import from_array

        # Extract cubed-specific kwargs.
        # Also ignores dask-specific kwargs that are passed in.
        # The passing of dask-specific kwargs to cubed should be eventually removed by deprecating them
        # as explicit arguments to xarray methods
        spec = kwargs.pop("spec", None)

        return from_array(
            data,
            chunks,
            spec=spec,
        )

    def compute(self, *data: "CubedArray", **kwargs) -> tuple[np.ndarray, ...]:
        from cubed import compute

        return compute(*data, **kwargs)

    @property
    def array_api(self) -> Any:
        from cubed import array_api

        return array_api

    def reduction(
        self,
        arr: "CubedArray",
        func: Callable,
        combine_func: Callable | None = None,
        aggregate_func: Callable | None = None,
        axis: int | Sequence[int] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
    ) -> "CubedArray":
        from cubed.core.ops import reduction

        return reduction(
            arr,
            func=func,
            combine_func=combine_func,
            aggegrate_func=aggregate_func,  # TODO fix the typo in argument name in cubed
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
        )

    def map_blocks(
        self,
        func: Callable,
        *args: Any,
        dtype: np.typing.DTypeLike | None = None,
        chunks: tuple[int, ...] | None = None,
        drop_axis: int | Sequence[int] | None = None,
        new_axis: int | Sequence[int] | None = None,
        **kwargs,
    ):
        from cubed.core.ops import map_blocks

        if drop_axis is None:
            # TODO should fix this upstream in cubed to match dask
            # see https://github.com/pydata/xarray/pull/7019#discussion_r1196729489
            drop_axis = []

        return map_blocks(
            func,
            *args,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
        )

    def blockwise(
        self,
        func: Callable,
        out_ind: Iterable,
        *args: Any,
        # can't type this as mypy assumes args are all same type, but blockwise args alternate types
        dtype: np.dtype | None = None,
        adjust_chunks: dict[Any, Callable] | None = None,
        new_axes: dict[Any, int] | None = None,
        align_arrays: bool = True,
        target_store=None,
        **kwargs,
    ):
        from cubed.core.ops import blockwise

        # TODO where to get the target_store kwarg from? Filter down from a blockwise call? Set as attribute on CubedManager?

        return blockwise(
            func,
            out_ind,
            *args,
            dtype=dtype,
            adjust_chunks=adjust_chunks,
            new_axes=new_axes,
            align_arrays=align_arrays,
            target_store=target_store,
            **kwargs,
        )

    def apply_gufunc(
        self,
        func: Callable,
        signature: str,
        *args: Any,
        axes: Sequence[tuple[int, ...]] | None = None,
        axis: int | None = None,
        keepdims: bool = False,
        output_dtypes: Sequence[np.typing.DTypeLike] | None = None,
        output_sizes: dict[str, int] | None = None,
        vectorize: bool | None = None,
        allow_rechunk: bool = False,
        **kwargs,
    ):
        if allow_rechunk:
            raise NotImplementedError(
                "cubed.apply_gufunc doesn't support allow_rechunk"
            )
        if keepdims:
            raise NotImplementedError("cubed.apply_gufunc doesn't support keepdims")

        from cubed import apply_gufunc

        return apply_gufunc(
            func,
            signature,
            *args,
            axes=axes,
            axis=axis,
            output_dtypes=output_dtypes,
            output_sizes=output_sizes,
            vectorize=vectorize,
            **kwargs,
        )

    def unify_chunks(
        self,
        *args: Any,  # can't type this as mypy assumes args are all same type, but dask unify_chunks args alternate types
        **kwargs,
    ) -> tuple[dict[str, T_NormalizedChunks], list["CubedArray"]]:
        from cubed.array_api import asarray
        from cubed.core import unify_chunks

        # Ensure that args are Cubed arrays. Note that we do this here and not in Cubed, following
        # https://numpy.org/neps/nep-0047-array-api-standard.html#the-asarray-asanyarray-pattern
        arginds = [
            (asarray(a) if ind is not None else a, ind) for a, ind in partition(2, args)
        ]
        array_args = [item for pair in arginds for item in pair]

        return unify_chunks(*array_args, **kwargs)

    def store(
        self,
        sources: Union["CubedArray", Sequence["CubedArray"]],
        targets: Any,
        **kwargs,
    ):
        """Used when writing to any backend."""
        from cubed.core.ops import store

        return store(
            sources,
            targets,
            **kwargs,
        )
