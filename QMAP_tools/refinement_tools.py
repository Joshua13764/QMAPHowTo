import shutil
from pathlib import Path
from typing import Callable, List

import numpy as np
import polars as pl
from joblib import delayed
from polars import DataFrame, LazyFrame
from tqdm_joblib import ParallelPbar

FACES: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"]


def extract_face(lf: LazyFrame, face: str | None,
                 columns: List[str], numb_workers: int = 4) -> List[np.ndarray]:

    lf_filtered: LazyFrame = lf.filter(
        pl.col("face") == face) if face is not None else lf

    def extract_func(face_lf: LazyFrame, column_name: str) -> np.ndarray:
        col_data: pl.DataFrame = face_lf.select(
            column_name, "i", "j").collect()

        values = col_data[column_name].to_numpy()
        i = col_data["i"].to_numpy()
        j = col_data["j"].to_numpy()

        arr = np.empty((i.max() + 1, j.max() + 1), dtype=values.dtype)
        arr[i, j] = values

        return arr

    extract_res = ParallelPbar(
        f"Extracting from face {face} into columns {columns}")(n_jobs=numb_workers)(
            delayed(extract_func)(lf_filtered, column) for column in columns
    )

    return list(extract_res)


def append_by_faces(
        target_lf: LazyFrame, export_folder: Path, col_name: str,
        process_face: Callable[[str], np.ndarray], skip_if_exists=False) -> None:

    assert export_folder.suffix == "", (
        f"export_folder should not have an extension, got: {export_folder}"
    )

    if export_folder.exists() and skip_if_exists:
        return
    elif export_folder.exists() and (not skip_if_exists):
        shutil.rmtree(export_folder)

    export_folder.mkdir(exist_ok=True, parents=True)

    for face in FACES:
        face_lf: LazyFrame = target_lf.filter(pl.col("face") == face)

        arr = process_face(face)

        face_df: DataFrame = face_lf.collect()

        values = arr[
            face_df["i"].to_numpy(),
            face_df["j"].to_numpy()
        ]

        face_df = face_df.with_columns(
            pl.Series(col_name, values)
        )

        face_df.write_parquet(export_folder / f"{face}.parquet")


def restrict_by_clipped_sigma(
        arr: np.ndarray, reduced_sigma=5) -> np.ndarray:
    arr_flat = arr.flatten()

    lower, upper = np.percentile(arr, [5, 95])
    masked_arr_flat = arr_flat[(arr_flat >= lower) & (arr_flat <= upper)]

    reduced_std: np.floating = masked_arr_flat.std()
    reduced_mean = np.floating = masked_arr_flat.mean()

    return np.clip(
        arr,
        reduced_mean - reduced_sigma * reduced_std,
        reduced_mean + reduced_sigma * reduced_std
    )


def restrict_by_clipped_sigma_log_space(
        arr: np.ndarray, reduced_sigma=5) -> np.ndarray:
    arr_flat = arr.flatten()

    lower, upper = np.percentile(arr[arr != 0], [5, 95])
    masked_arr_flat = arr_flat[(arr_flat >= lower) & (arr_flat <= upper)]

    reduced_std: np.floating = np.log(masked_arr_flat).std()
    reduced_mean = np.floating = np.log(masked_arr_flat).mean()

    return np.clip(
        arr,
        np.exp(reduced_mean - reduced_sigma * reduced_std),
        np.exp(reduced_mean + reduced_sigma * reduced_std)
    )
