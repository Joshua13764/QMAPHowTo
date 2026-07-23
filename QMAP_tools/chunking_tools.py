import shutil
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from polars import DataFrame, Expr, LazyFrame
from tqdm import tqdm
from tqdm_joblib import ParallelPbar
from .qcube_chunk import QCubeChunk

class ChunkingTools:
    @staticmethod
    def extract_chunks(lf: LazyFrame, chunk: QCubeChunk,
                       columns: List[str], numb_workers: int = 4,
                       verbose=False) -> List[np.ndarray]:

        lf_filtered: LazyFrame = chunk.filter_lf(lf)

        def extract_func(chunk_lf: LazyFrame, column_name: str) -> np.ndarray:
            col_data: pl.DataFrame = chunk_lf.select(
                column_name, "i", "j").collect()

            values = col_data[column_name].to_numpy()
            i = col_data["i"].to_numpy()
            j = col_data["j"].to_numpy()

            arr = np.empty(
                (i.max() - i.min() + 1,
                 j.max() - j.min() + 1),
                dtype=values.dtype)
            arr[i - i.min(), j - j.min()] = values

            return arr

        parallel = (
            ParallelPbar(
                f"Extracting from chunk {chunk} into columns {columns}"
            )(n_jobs=numb_workers)
            if verbose
            else Parallel(n_jobs=numb_workers)
        )

        extract_res = parallel(
            delayed(extract_func)(lf_filtered, column)
            for column in columns
        )

        return list(extract_res)

    @staticmethod
    def extract_downsampled_chunks(
        lf: LazyFrame,
        chunk: QCubeChunk,
        columns: List[str],
        downscale_factor: int,
        verbose : bool = False
    ) -> List[np.ndarray]:
        """
        Extract downsampled arrays from a chunk by aggregating over
        (face, i // downscale_factor, j // downscale_factor).

        Parameters
        ----------
        lf
            Source LazyFrame.
        chunk
            Chunk to extract.
        columns
            Columns to extract.
        downscale_factor
            Integer factor by which to downsample each spatial dimension.
            A value of 1 performs no downsampling.

        Returns
        -------
        List[np.ndarray]
            One downsampled array per requested column.
        """
        if downscale_factor < 1:
            raise ValueError("downscale_factor must be >= 1")

        lf_filtered = chunk.filter_lf(lf)

        def extract_column(column: str) -> np.ndarray:
            df = (
                lf_filtered
                .with_columns(
                    (pl.col("i") // downscale_factor).alias("i"),
                    (pl.col("j") // downscale_factor).alias("j"),
                )
                .select("face", "i", "j", column)
                .group_by("face", "i", "j")
                .agg(pl.col(column).mean())
                .collect(engine="streaming")
            )

            values = df[column].to_numpy()
            i = df["i"].to_numpy()
            j = df["j"].to_numpy()

            i_min = i.min()
            j_min = j.min()

            arr = np.empty(
                (
                    i.max() - i_min + 1,
                    j.max() - j_min + 1,
                ),
                dtype=values.dtype,
            )

            arr[i - i_min, j - j_min] = values
            return arr

        return [
            extract_column(column)
            for column in (
                    tqdm(
                    columns,
                    desc=f"Extracting downsampled chunk {chunk} with scale factor {downscale_factor}",
                ) if verbose else columns
            )
        ]

    @staticmethod
    def append_by_chunks(
            target_lf: LazyFrame, export_folder: Path, col_name: str,
            process_chunk: Callable[[QCubeChunk], np.ndarray],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=1),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk_lf: LazyFrame = chunk.filter_lf(target_lf)

            arr: np.ndarray = process_chunk(chunk)

            chunk_df: DataFrame = chunk_lf.select("i", "j").collect()

            values: np.ndarray = arr[
                chunk_df["i"].to_numpy() - chunk.i_min,
                chunk_df["j"].to_numpy() - chunk.j_min
            ]

            chunk_lf = chunk_lf.with_columns(
                pl.Series(col_name, values)
            )

            chunk_lf.sink_parquet(
                export_folder / f"{chunk.short_name}.parquet",
                engine="streaming"
            )

    @staticmethod
    def join_full_with_aggs(
            export_folder: Path,
            full_db: pl.LazyFrame,
            aggs_to_join_with: Dict[Tuple[str], pl.DataFrame],  # Join on : df
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        def process_chunk(chunk: QCubeChunk) -> None:
            full_chunked_df: pl.LazyFrame = chunk.filter_lf(full_db)

            reduce(
                lambda left, right:
                    left.join(
                        right[1].lazy(),
                        on=list(right[0]),
                        how="left",
                        coalesce=True,
                    ),
                aggs_to_join_with.items(),
                full_chunked_df,
            ).sink_parquet(
                export_folder / f"{chunk.short_name}.parquet",
                engine="streaming"
            )

        for chunk in tqdm(chunks, desc="Joining full with aggs"):
            process_chunk(chunk)

    @staticmethod
    def join_full_with_agg(
            export_folder: Path,
            full_db: pl.LazyFrame,
            agg_db: pl.DataFrame,
            join_on: List[str] = ["boulder_id"],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False, n_jobs=4) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        def process_chunk(chunk: QCubeChunk) -> None:
            full_chunked_df: DataFrame = chunk.filter_lf(full_db).collect()
            full_chunked_df.join(
                agg_db,
                on=join_on,
                how="inner"
            ).write_parquet(
                export_folder / f"{chunk.short_name}.parquet"
            )

        ParallelPbar("Joining full with agg")(n_jobs=n_jobs)(
            delayed(process_chunk)(chunk) for chunk in chunks
        )

    @staticmethod
    def join_in_chunks(
            export_folder: Path,
            # Left join so the first one needs to be full
            lfs_to_join: List[LazyFrame],
            join_on: List[str] = ["i", "j", "face"],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=3),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        if lfs_to_join[0].filter(
                pl.col("i") == 1).collect().height != 8192 * 6:
            print("Cannot do merge as first input lf is not full for i, j and face")
            return

        def process_chunk(chunk) -> None:
            filtered: List[pl.LazyFrame] = [
                chunk.filter_lf(df)
                for df in lfs_to_join
            ]

            combined: LazyFrame = reduce(
                lambda left, right: left.join(
                    right,
                    on=join_on,
                    how="left",
                    coalesce=True,
                ),
                filtered,
            )

            combined.sink_parquet(
                export_folder / f"{chunk.short_name}.parquet")

        for chunk in tqdm(chunks, desc="Joining"):
            process_chunk(chunk)

    @staticmethod
    def agg_in_slices(
            export_df_path: Path,
            lf_to_agg: LazyFrame,
            agg_group: str,
            agg_exprs: List[Expr],
            slice_size: int = 1_000,
            skip_if_exists=False, n_jobs=4) -> None:

        if export_df_path.exists() and skip_if_exists:
            return

        export_df_path.parent.mkdir(exist_ok=True, parents=True)
        groups: pl.Series = lf_to_agg.group_by(
            agg_group).agg().collect()[agg_group].sort()
        print(f"Found {len(groups)} groups")

        group_slices: List[pl.Series] = [
            groups.slice(i, slice_size)
            for i in range(0, len(groups), slice_size)
        ]

        def process_group_slice(group_slice: pl.Series) -> DataFrame:
            agg_data: DataFrame = lf_to_agg.filter(pl.col(agg_group).is_in(
                group_slice.implode())).collect().group_by(
                    agg_group
            ).agg(*agg_exprs)

            return agg_data

        agg_data_dfs: List[DataFrame | None] = list(ParallelPbar("Joining")(n_jobs=n_jobs)(
            delayed(process_group_slice)(group_slice) for group_slice in group_slices
        ))

        merged_df: pl.DataFrame = pl.concat(agg_data_dfs)
        merged_df.write_parquet(export_df_path)

    @staticmethod
    def bulk_append_by_chunks(
            target_lf: LazyFrame, export_folder: Path, col_names: List[str],
            process_chunk: Callable[[QCubeChunk], List[np.ndarray]],
            chunks: List[QCubeChunk] = QCubeChunk.generate(depth=1),
            skip_if_exists=False) -> None:

        assert export_folder.suffix == "", (
            f"export_folder should not have an extension, got: {export_folder}"
        )

        if export_folder.exists() and skip_if_exists:
            return
        elif export_folder.exists() and (not skip_if_exists):
            shutil.rmtree(export_folder)

        export_folder.mkdir(exist_ok=True, parents=True)

        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk_lf: LazyFrame = chunk.filter_lf(target_lf)
            chunk_df: DataFrame = chunk_lf.collect()

            for arr, col_name in zip(process_chunk(chunk), col_names):

                values: np.ndarray = arr[
                    chunk_df["i"].to_numpy() - chunk.i_min,
                    chunk_df["j"].to_numpy() - chunk.j_min
                ]

                chunk_df = chunk_df.with_columns(
                    pl.Series(col_name, values)
                )

            chunk_df.write_parquet(
                export_folder / f"{chunk.short_name}.parquet"
            )
