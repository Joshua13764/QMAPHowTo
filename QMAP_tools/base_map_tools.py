from typing import Dict, List, Tuple

import cv2
import numpy as np
import polars as pl
from polars import LazyFrame

from .qcube_chunk import QCubeChunk
from .chunking_tools import ChunkingTools

class BaseMapTools:

    @staticmethod
    def extract_column_as_equirectangular(
            df: LazyFrame, column_name: str, output_resolution: Tuple[int, int] = (1024, 512)):
        return BaseMapTools.cubemap_to_equirectangular(
            faces=BaseMapTools.extract_column_as_faces(df, column_name),
            output_resolution=output_resolution
        )
    
    @staticmethod
    def extract_column_as_cube_net(
            df: LazyFrame, column_name: str, target_face_size: int = 512) -> np.ndarray:
        return BaseMapTools.faces_to_cubemap_net(
            faces=BaseMapTools.extract_column_as_faces(df, column_name, target_face_size = target_face_size)
        )

    @staticmethod
    def extract_column_as_faces(df: LazyFrame, column_name: str, chunk_depth: int = 0,
                                target_face_size: int = 1024, source_face_size: int = 8192) -> Dict[str, np.ndarray]:
        faces = {}

        for chunk in QCubeChunk.generate(
                chunk_depth, face_size=source_face_size):

            # Scale chunk placement from source face coordinates -> target face
            # coordinates
            i_min = chunk.i_min * target_face_size // source_face_size
            i_max = chunk.i_max * target_face_size // source_face_size
            j_min = chunk.j_min * target_face_size // source_face_size
            j_max = chunk.j_max * target_face_size // source_face_size

            # Final tile size in the output face
            target_height = i_max - i_min
            target_width = j_max - j_min

            # Chunk size in the source face
            source_height = chunk.i_max - chunk.i_min
            source_width = chunk.j_max - chunk.j_min

            downscale_factor = max(
                1,
                min(
                    source_height // target_height,
                    source_width // target_width,
                ),
            )

            r = ChunkingTools.extract_downsampled_chunks(
                df,
                chunk,
                [column_name],
                downscale_factor=downscale_factor
            )[0]

            # print(type(r), r.dtype, r.shape)
            # print(r[:5] if r.ndim == 1 else r[:2, :2])

            # Resize tile directly to its final size
            r = cv2.resize(
                r,
                (j_max - j_min, i_max - i_min),
                interpolation=cv2.INTER_AREA
            )

            if chunk.face not in faces:
                faces[chunk.face] = np.empty(
                    (target_face_size, target_face_size),
                    dtype=r.dtype
                )

            faces[chunk.face][i_min:i_max, j_min:j_max] = r

        faces = {
            "posx": faces["posx"],
            "negx": faces["negx"],
            "posy": faces["posy"],
            "negy": faces["negy"],
            "posz": faces["posz"],
            "negz": faces["negz"],
        }

        return faces

    @staticmethod
    def cubemap_to_equirectangular(
        faces: dict[str, np.ndarray],
        output_resolution: tuple[int, int]
    ) -> np.ndarray:
        """
        Inverse of PANToCubemap.sample_face_roi_simple_super_sample()

        Input faces must use the same convention:
            posx: ( 1, V,-U)
            negx: (-1, V, U)
            posy: ( U, 1,-V)
            negy: ( U,-1, V)
            posz: ( U, V, 1)
            negz: (-U, V,-1)

        output_resolution:
            (width,height)

        Supports:
            HxW
            HxWxC
        """

        out_w, out_h = output_resolution

        face0 = faces["posx"]

        if face0.ndim == 2:
            result = np.empty(
                (out_h, out_w),
                dtype=face0.dtype
            )
        else:
            result = np.empty(
                (out_h, out_w, face0.shape[2]),
                dtype=face0.dtype
            )

        # -----------------------------
        # Equirectangular pixel grid
        # -----------------------------

        px = np.arange(out_w, dtype=np.float32)
        py = np.arange(out_h, dtype=np.float32)

        X, Y = np.meshgrid(px, py)

        # Same convention as forward:
        #
        # lon = atan2(x,z)
        # lat = asin(y)

        lon = (X / (out_w - 1) - 0.5) * 2 * np.pi
        lat = (0.5 - Y / (out_h - 1)) * np.pi

        # longitude/latitude -> direction vector

        dir_x = np.cos(lat) * np.sin(lon)
        dir_y = np.sin(lat)
        dir_z = np.cos(lat) * np.cos(lon)

        ax = np.abs(dir_x)
        ay = np.abs(dir_y)
        az = np.abs(dir_z)

        def sample_face(name, U, V, mask):

            if not np.any(mask):
                return

            h, w = faces[name].shape[:2]

            # [-1,1] -> pixel coordinates
            mapx = ((U + 1) * 0.5) * (w - 1)
            mapy = ((V + 1) * 0.5) * (h - 1)

            sampled = cv2.remap(
                faces[name],
                mapx.astype(np.float32),
                mapy.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )

            result[mask] = sampled[mask]

        eps = 1e-8

        # -----------------------------
        # +X
        # -----------------------------

        mask = (
            (ax >= ay) &
            (ax >= az) &
            (dir_x > 0)
        )

        U = -dir_z / np.maximum(ax, eps)
        V = dir_y / np.maximum(ax, eps)

        sample_face("posx", U, V, mask)

        # -----------------------------
        # -X
        # -----------------------------

        mask = (
            (ax >= ay) &
            (ax >= az) &
            (dir_x <= 0)
        )

        U = dir_z / np.maximum(ax, eps)
        V = dir_y / np.maximum(ax, eps)

        sample_face("negx", U, V, mask)

        # -----------------------------
        # +Y
        # -----------------------------

        mask = (
            (ay >= ax) &
            (ay >= az) &
            (dir_y > 0)
        )

        U = dir_x / np.maximum(ay, eps)
        V = -dir_z / np.maximum(ay, eps)

        sample_face("posy", U, V, mask)

        # -----------------------------
        # -Y
        # -----------------------------

        mask = (
            (ay >= ax) &
            (ay >= az) &
            (dir_y <= 0)
        )

        U = dir_x / np.maximum(ay, eps)
        V = dir_z / np.maximum(ay, eps)

        sample_face("negy", U, V, mask)

        # -----------------------------
        # +Z
        # -----------------------------

        mask = (
            (az >= ax) &
            (az >= ay) &
            (dir_z > 0)
        )

        U = dir_x / np.maximum(az, eps)
        V = dir_y / np.maximum(az, eps)

        sample_face("posz", U, V, mask)

        # -----------------------------
        # -Z
        # -----------------------------

        mask = (
            (az >= ax) &
            (az >= ay) &
            (dir_z <= 0)
        )

        U = -dir_x / np.maximum(az, eps)
        V = dir_y / np.maximum(az, eps)

        sample_face("negz", U, V, mask)

        return result

    @staticmethod
    def faces_to_cubemap_net(
        faces: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Packs cubemap faces into the same horizontal cross layout:

                negy
        negx     posz     posx     negz
                posy

        Matches:

            negy: (1,0)
            posy: (1,2)
            negx: (0,1)
            posz: (1,1)
            posx: (2,1)
            negz: (3,1)

        """

        face = faces["posx"]

        h, w = face.shape[:2]

        if face.ndim == 2:
            cubemap = np.zeros(
                (3 * h, 4 * w),
                dtype=face.dtype
            )
        else:
            cubemap = np.zeros(
                (3 * h, 4 * w, face.shape[2]),
                dtype=face.dtype
            )

        # x, y positions in the net

        layout = {
            "negy": (1, 0),
            "posy": (1, 2),
            "negx": (0, 1),
            "posz": (1, 1),
            "posx": (2, 1),
            "negz": (3, 1),
        }

        for name, (cx, cy) in layout.items():
            cubemap[
                cy * h:(cy + 1) * h,
                cx * w:(cx + 1) * w
            ] = faces[name]

        return cubemap
