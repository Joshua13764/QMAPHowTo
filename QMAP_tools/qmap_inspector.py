from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame
from pathlib import Path
from matplotlib.gridspec import GridSpec

@dataclass(frozen=True, kw_only=True)
class QMAPInspector():
    full_db: LazyFrame = field(
        default_factory=lambda: pl.scan_parquet(r"QMAP_data\pass_02v3"))
    agg_db: DataFrame = field(
        default_factory=lambda: pl.read_parquet(r"QMAP_data\boulder_id.parquet"))

    def get_flat_region(
        self,
        face: str,
        i_range: tuple[int, int],
        j_range: tuple[int, int],
        column_names: list[str],
        custom_lf: LazyFrame | None = None,
        max_pixels: int = 4 * 4096 ** 2,
    ) -> DataFrame:
        min_i, max_i = i_range
        min_j, max_j = j_range

        expected_size = (max_i - min_i + 1) * (max_j - min_j + 1) * len(column_names)

        if expected_size > max_pixels:
            raise ValueError(
                f"Requested flat region is too large: "
                f"{expected_size:,} pixels "
                f"({max_i - min_i + 1} x {max_j - min_j + 1}). "
                f"Maximum allowed is {max_pixels:,} pixels."
            )

        return (
            (self.full_db if custom_lf is None else custom_lf)
            .filter(
                (pl.col("face") == face)
                & (pl.col("i") >= min_i)
                & (pl.col("i") <= max_i)
                & (pl.col("j") >= min_j)
                & (pl.col("j") <= max_j)
            )
            .select(column_names)
            .collect()
        )

    def get_full_df_around_boulder(
        self,
        boulder_id: int,
        column_names: list[str],
        zoom_factor: float = 1.5,
        custom_lf: LazyFrame | None = None,
    ) -> DataFrame:
        lf = self.full_db if custom_lf is None else custom_lf

        boulder_df = lf.filter(
            pl.col("boulder_id") == boulder_id
        ).collect()

        min_i, max_i = int(boulder_df["i"].min()), int(boulder_df["i"].max())
        min_j, max_j = int(boulder_df["j"].min()), int(boulder_df["j"].max())
        face = boulder_df["face"][0]

        center_i = (min_i + max_i) / 2
        center_j = (min_j + max_j) / 2

        half_width = (max_i - min_i) * zoom_factor / 2
        half_height = (max_j - min_j) * zoom_factor / 2

        i_range = (
            int(center_i - half_width),
            int(center_i + half_width),
        )
        j_range = (
            int(center_j - half_height),
            int(center_j + half_height),
        )

        return self.get_flat_region(
            face=face,
            i_range=i_range,
            j_range=j_range,
            custom_lf=custom_lf,
            column_names = column_names
        )

    def render_flat_region(
        self,
        face: str,
        i_range: tuple[int, int],
        j_range: tuple[int, int],
        column_name: str,
        custom_lf: LazyFrame | None = None,
    ) -> np.ndarray:
        region_df: DataFrame = self.get_flat_region(
            face, i_range, j_range,
            column_names=["i", "j", "face", column_name],
            custom_lf=custom_lf
            )

        i_min, i_max = i_range
        j_min, j_max = j_range

        arr = np.full(
            (i_max - i_min + 1, j_max - j_min + 1),
            np.nan,
            dtype=np.float32,
        )

        arr[
            region_df["i"].to_numpy() - i_min,
            region_df["j"].to_numpy() - j_min,
        ] = region_df[column_name].to_numpy()

        return arr

    def render_column_by_face(
        self,
        render_lf: LazyFrame,
        column_name: str,
        custom_lf: LazyFrame | None = None,
    ) -> dict[str, np.ndarray]:
        rendered_faces: dict[str, np.ndarray] = {}

        faces = (
            render_lf
            .select(pl.col("face").unique())
            .collect()["face"]
            .to_list()
        )

        for face in faces:
            bounds = (
                render_lf
                .filter(pl.col("face") == face)
                .select(
                    pl.col("i").min().alias("min_i"),
                    pl.col("i").max().alias("max_i"),
                    pl.col("j").min().alias("min_j"),
                    pl.col("j").max().alias("max_j"),
                )
                .collect()
            )

            i_range = (
                int(bounds["min_i"][0]),
                int(bounds["max_i"][0]),
            )
            j_range = (
                int(bounds["min_j"][0]),
                int(bounds["max_j"][0]),
            )

            rendered_faces[face] = self.render_flat_region(
                face=face,
                i_range=i_range,
                j_range=j_range,
                column_name=column_name,
                custom_lf=custom_lf,
            )

        return rendered_faces

    def render_column_around_boulder(
        self,
        boulder_id: int,
        zoom_factor: float,
        column_name: str,
        custom_lf: LazyFrame | None = None,
    ) -> np.ndarray:
        boulder_area_df: DataFrame = self.get_full_df_around_boulder(
            boulder_id,
            column_names=["i", "j", "face", column_name],
            zoom_factor = zoom_factor,
            custom_lf=custom_lf
        )

        i_range = (
            int(boulder_area_df["i"].min()),
            int(boulder_area_df["i"].max()),
        )
        j_range = (
            int(boulder_area_df["j"].min()),
            int(boulder_area_df["j"].max()),
        )

        return self.render_flat_region(
            face=boulder_area_df["face"][0],
            i_range=i_range,
            j_range=j_range,
            column_name=column_name,
            custom_lf=custom_lf,
        )
    
    def render_multi_plot(
        self,
        boulder_id: int,
        left_view_column: str,
        right_view_column: str = "32bit_reflectance",
        zoom_factor: float = 1.5,
        fig_size : tuple[float, float] = (10, 4),
        fig_export_path: Path | None = None,
        custom_lf: LazyFrame | None = None,
    ) -> None:
        left = self.render_column_around_boulder(
            boulder_id, zoom_factor, left_view_column, custom_lf
        )
        right = self.render_column_around_boulder(
            boulder_id, zoom_factor, right_view_column, custom_lf
        )

        # Get metadata
        metadata_df = (
            self.agg_db
            .filter(pl.col("boulder_id") == boulder_id)
            .with_columns(pl.lit(zoom_factor).alias("zoom_factor"))
        )

        # Format metadata for the figure
        if metadata_df.height > 0:
            row = metadata_df.row(0, named=True)

            items = []
            for key, value in row.items():
                if isinstance(value, (float, np.floating)):
                    value = f"{value:.3f}"
                items.append(f"{key}: {value}")

            n_per_line = 9
            metadata_text = "\n".join(
                ", ".join(items[i:i + n_per_line])
                for i in range(0, len(items), n_per_line)
            )
        else:
            metadata_text = (
                f"boulder_id: {boulder_id}, "
                f"zoom_factor: {zoom_factor:.3f}"
            )

        # Create figure
        fig = plt.figure(figsize=fig_size)
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[4, 1.5],
        )

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[1, :])

        # Left image
        im_left = ax_left.imshow(left, origin="lower")
        ax_left.set_title(left_view_column)
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        fig.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)

        # Right image
        im_right = ax_right.imshow(right, origin="lower")
        ax_right.set_title(right_view_column)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        fig.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)

        # Metadata
        ax_text.axis("off")
        ax_text.text(
            0.01,
            0.99,
            metadata_text,
            transform=ax_text.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
        )

        fig.tight_layout()

        if fig_export_path is None:
            plt.show()
        else:
            fig.savefig(fig_export_path, bbox_inches="tight")
            plt.close(fig)

    def get_column_extrema(
        self,
        column: str,
        custom_lf : LazyFrame | None = None
    ) -> tuple[float, float]:
        """
        Get column minimum and maximum using streaming.

        Returns:
            (min_value, max_value)
        """
        extrema = (
            (self.full_db if custom_lf is None else custom_lf)
            .select(
                [
                    pl.col(column).min().alias("min"),
                    pl.col(column).max().alias("max"),
                ]
            )
            .collect(engine="streaming")
        )

        return extrema["min"][0], extrema["max"][0]

    def get_column_hist(
        self,
        column: str,
        bin_range: tuple[float, float],
        bin_number: int = 100,
        custom_lf : LazyFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear histogram using streaming Polars aggregation.

        Returns:
            counts: np.ndarray of bin counts
            bins: np.ndarray of bin edges
        """
        min_val, max_val = bin_range

        if max_val <= min_val:
            raise ValueError("bin_range must satisfy max > min")

        bins: np.ndarray = np.linspace(min_val, max_val, bin_number + 1)
        bin_width: float = (max_val - min_val) / bin_number

        hist: DataFrame = (
            (self.full_db if custom_lf is None else custom_lf)
            .filter(
                pl.col(column).is_between(min_val, max_val, closed="left")
            )
            .select(
                (
                    ((pl.col(column) - min_val) / bin_width)
                    .floor()
                    .cast(pl.UInt32)
                    .alias("bin")
                )
            )
            .group_by("bin")
            .len(name="count")
            .collect(engine="streaming")
        )

        counts: np.ndarray = np.zeros(bin_number, dtype=np.int64)

        counts[
            hist["bin"].to_numpy()
        ] = hist["count"].to_numpy()

        return counts, bins

    def get_column_hist_log(
        self,
        column: str,
        bin_range: tuple[float, float],
        bin_number: int,
        custom_lf : LazyFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Logarithmic histogram using streaming Polars aggregation.

        Returns:
            counts: np.ndarray of bin counts
            bins: np.ndarray of logarithmic bin edges
        """
        min_val, max_val = bin_range

        if min_val <= 0:
            raise ValueError("Log histogram requires min_val > 0")
        if max_val <= min_val:
            raise ValueError("bin_range must satisfy max > min")

        bins = np.logspace(
            np.log10(min_val),
            np.log10(max_val),
            bin_number + 1,
        )

        log_min = np.log10(min_val)
        log_bin_width = (
            np.log10(max_val) - log_min
        ) / bin_number

        hist = (
            (self.full_db if custom_lf is None else custom_lf)
            .filter(
                pl.col(column).is_between(min_val, max_val, closed="left")
            )
            .select(
                (
                    (
                        (pl.col(column).log10() - log_min)
                        / log_bin_width
                    )
                    .floor()
                    .cast(pl.UInt32)
                    .alias("bin")
                )
            )
            .group_by("bin")
            .len(name="count")
            .collect(engine="streaming")
        )

        counts = np.zeros(bin_number, dtype=np.int64)

        counts[
            hist["bin"].to_numpy()
        ] = hist["count"].to_numpy()

        return counts, bins
    
    def get_column_hist2d(
        self,
        columns: tuple[str, str],
        bin_ranges: tuple[tuple[float, float], tuple[float, float]],
        bin_numbers: tuple[int, int] = (100, 100),
        custom_lf: LazyFrame | None = None,
        weight_col: str | None = "area",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D linear histogram using streaming Polars aggregation.

        If weight_col is provided, bins contain the sum of weights
        instead of the number of entries.

        Returns:
            counts: np.ndarray of shape (x_bins, y_bins)
            x_bins: np.ndarray of x-axis bin edges
            y_bins: np.ndarray of y-axis bin edges
        """

        x_col, y_col = columns
        (x_min, x_max), (y_min, y_max) = bin_ranges
        x_bins_num, y_bins_num = bin_numbers

        if x_max <= x_min:
            raise ValueError("x bin_range must satisfy max > min")

        if y_max <= y_min:
            raise ValueError("y bin_range must satisfy max > min")

        x_bins = np.linspace(x_min, x_max, x_bins_num + 1)
        y_bins = np.linspace(y_min, y_max, y_bins_num + 1)

        x_width = (x_max - x_min) / x_bins_num
        y_width = (y_max - y_min) / y_bins_num

        lf = (
            self.full_db if custom_lf is None else custom_lf
        )

        select_exprs = [
            (
                ((pl.col(x_col) - x_min) / x_width)
                .floor()
                .cast(pl.UInt32)
                .alias("x_bin")
            ),
            (
                ((pl.col(y_col) - y_min) / y_width)
                .floor()
                .cast(pl.UInt32)
                .alias("y_bin")
            ),
        ]

        if weight_col is not None:
            select_exprs.append(pl.col(weight_col).alias("weight"))

        hist = (
            lf
            .filter(
                pl.col(x_col).is_between(x_min, x_max, closed="left")
                &
                pl.col(y_col).is_between(y_min, y_max, closed="left")
            )
            .select(select_exprs)
            .group_by(["x_bin", "y_bin"])
            .agg(
                pl.col("weight").sum().alias("count")
                if weight_col is not None
                else pl.len().alias("count")
            )
            .collect(engine="streaming")
        )

        counts = np.zeros(
            (x_bins_num, y_bins_num),
            dtype=np.float64 if weight_col is not None else np.int64
        )

        counts[
            hist["x_bin"].to_numpy(),
            hist["y_bin"].to_numpy()
        ] = hist["count"].to_numpy()

        return counts, x_bins, y_bins