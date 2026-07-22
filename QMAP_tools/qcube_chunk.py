from dataclasses import dataclass
from itertools import product
from typing import List
import polars as pl
from polars import LazyFrame

FACES: List[str] = ["posx", "posy", "posz", "negx", "negy", "negz"]


@dataclass(frozen=True)
class QCubeChunk():
    face: str
    i_min: int
    j_min: int
    length: int

    @property
    def i_max(self) -> int:
        return self.i_min + self.length

    @property
    def j_max(self) -> int:
        return self.j_min + self.length

    @property
    def x_range(self) -> tuple[float, float]:
        return (
            self.i_min / 8192,
            self.i_max / 8192
        )

    @property
    def y_range(self) -> tuple[float, float]:
        return (
            self.j_min / 8192,
            self.j_max / 8192
        )

    def filter_lf(self, lf: LazyFrame) -> LazyFrame:
        return lf.filter(
            pl.col("face") == self.face,

            pl.col("i") >= self.i_min,
            pl.col("i") < self.i_max,

            pl.col("j") >= self.j_min,
            pl.col("j") < self.j_max
        )

    @staticmethod
    def generate(depth: int, face_size: int = 8192) -> list["QCubeChunk"]:
        divisions: int = 2 ** depth
        chunk_size: int = face_size // divisions

        return [
            QCubeChunk(
                face=face,
                i_min=i_block * chunk_size,
                j_min=j_block * chunk_size,
                length=chunk_size,
            )
            for face, i_block, j_block in product(
                FACES,
                range(divisions),
                range(divisions),
            )
        ]

    @property
    def short_name(self) -> str:
        return f"QCubeChunk_face_{self.face}_length_{self.length}_i_{self.i_min}_j_{self.j_min}"

    def __repr__(self) -> str:
        return (
            f"QCubeChunk("
            f"face={self.face!r}, "
            f"i=[{self.i_min}, {self.i_max}), "
            f"j=[{self.j_min}, {self.j_max}), "
            f"length={self.length}"
            f")"
        )
