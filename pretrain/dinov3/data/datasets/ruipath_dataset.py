"""RuiPath datasets for individual image files and HDF5 patch bundles."""

import csv
import hashlib
import operator
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .extended import ExtendedVisionDataset


class RuiPathPlainDataset(ExtendedVisionDataset):
    """Read RGB images from a text file containing one image path per line.

    Returns (transformed_image, transformed_target),
    with an initial placeholder target of zero. ``verify`` is retained for
    compatibility; images are validated when read.
    """

    def __init__(
        self,
        meta_file: str,
        verify: bool = False,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.meta_txt = Path(meta_file).resolve()
        with self.meta_txt.open() as stream:
            self.image_paths = [line.strip() for line in stream]

    def get_target(self, index: int) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        try:
            with Image.open(self.image_paths[index]) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot read image for sample {index}") from exc
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class RuiPathH5Dataset(ExtendedVisionDataset):
    """Read RGB patches using ``h5_path,patch_count,cum_sum`` metadata.

    ``patch_count`` is a sampling quota, not necessarily the H5 length.
    For N stored patches and M requested samples, logical sample i selects
    from [floor(i*N/M), floor((i+1)*N/M)). Empty intervals when M > N reuse
    the patch at their lower bound. Equal counts give an identity mapping.

    Sampling is deterministic per H5 path in the manifest and logical index, independent
    of worker, rank and access order; it does not resample between epochs.
    The H5 ``patches`` dataset is authoritative; ``patch_total`` is not needed.
    Files are opened per access, so no HDF5 handles are shared between workers.
    Returns (transformed_image, transformed_target, "row-logical_index").
    """

    def __init__(
        self,
        meta_csv: str,
        root: str,
        verify: bool = False,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.root = Path(root).expanduser().resolve()
        self.meta_csv = Path(meta_csv).expanduser().resolve()
        self.h5_paths = []
        self.patch_counts = []
        cumulative_counts = []
        total = 0
        delimiter = "\t" if self.meta_csv.suffix.lower() == ".tsv" else ","
        with self.meta_csv.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            required = {"h5_path", "patch_count", "cum_sum"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{self.meta_csv}: required columns are {sorted(required)}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    path = (row["h5_path"] or "").strip()
                    count = int(row["patch_count"])
                    cumulative = int(row["cum_sum"])
                    if not path or count < 0:
                        raise ValueError("h5_path must be nonempty and patch_count must be nonnegative")
                    total += count
                    if cumulative != total:
                        raise ValueError(f"cum_sum is {cumulative}, expected {total}")
                    if total > min(sys.maxsize, np.iinfo(np.int64).max):
                        raise ValueError("total sample count exceeds the supported index range")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{self.meta_csv}: invalid metadata at row {row_number}: {exc}") from exc
                self.h5_paths.append(path)
                self.patch_counts.append(count)
                cumulative_counts.append(total)
        if total == 0:
            raise ValueError(f"{self.meta_csv}: at least one sample is required")
        self.cumulative_counts = np.asarray(cumulative_counts, dtype=np.int64)

        if verify:
            for path, count in zip(self.h5_paths, self.patch_counts):
                if count == 0:
                    continue
                full_path = self.root / path
                try:
                    with h5py.File(full_path, "r") as bundle:
                        self._get_patches(bundle)
                except Exception as exc:
                    raise RuntimeError(f"Invalid H5 bundle {full_path}: {exc}") from exc

    @staticmethod
    def _get_patches(bundle: h5py.File) -> h5py.Dataset:
        patches = bundle["patches"]
        if not isinstance(patches, h5py.Dataset) or patches.ndim not in (3, 4):
            raise ValueError("patches must be an array of shape [N,H,W] or [N,H,W,C]")
        if patches.shape[0] == 0:
            raise ValueError("patches is empty but the sampling quota is positive")
        return patches

    @staticmethod
    def _sample_patch_index(path: str, index: int, actual_count: int, quota: int) -> int:
        # Python integer arithmetic avoids both float rounding and uint16 overflow.
        low = index * actual_count // quota
        high = (index + 1) * actual_count // quota
        if high <= low + 1:
            return low
        digest = hashlib.blake2b(f"{path}\0{index}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        return int(rng.integers(low, high))  # high is exclusive; do not subtract 1

    def __len__(self) -> int:
        return int(self.cumulative_counts[-1])

    def get_target(self, index: int) -> int:
        return 0

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError(f"Sample index {index} is outside [0, {len(self)})")
        h5_index = int(np.searchsorted(self.cumulative_counts, index, side="right"))
        quota = self.patch_counts[h5_index]
        previous_count = int(self.cumulative_counts[h5_index]) - quota
        logical_index = index - previous_count
        path = self.h5_paths[h5_index]
        full_path = self.root / path
        patch_index = None
        try:
            with h5py.File(full_path, "r") as bundle:
                patches = self._get_patches(bundle)
                patch_index = self._sample_patch_index(path, logical_index, patches.shape[0], quota)
                image = Image.fromarray(patches[patch_index]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read sample {index}: file={full_path}, row={h5_index}, "
                f"logical_index={logical_index}, patch_index={patch_index}: {exc}"
            ) from exc
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, f"{h5_index}-{logical_index}"
