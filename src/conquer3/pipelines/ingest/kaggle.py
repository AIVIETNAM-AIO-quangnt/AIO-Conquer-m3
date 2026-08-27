"""Downloads the PaySim1 CSV from Kaggle, for local dev without a pre-placed file.

Colab uses ``userdata`` secrets instead (see README). PaySim1 is a public dataset,
so ``KAGGLE_USERNAME``/``KAGGLE_KEY`` are optional -- ``kagglehub`` falls back to an
anonymous download, and (confirmed empirically) already extracts the zip it fetches
before handing back a directory of plain files. Credentials only matter for private
or rate-limited datasets.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from conquer3.config.settings import KaggleSettings, get_settings

__all__ = ["download_paysim_csv", "extract_single_csv"]

_CSV_GLOB = "*.csv"


def download_paysim_csv(dest: Path | str, *, kaggle: KaggleSettings | None = None) -> Path:
    """Downloads the configured Kaggle dataset and copies its CSV to ``dest``.

    Raises if the dataset doesn't contain exactly one CSV -- PaySim1 does, and a
    surprise archive layout is a data-quality problem worth failing loudly on, not
    guessing through.
    """
    import kagglehub

    settings = kaggle if kaggle is not None else get_settings().kaggle
    dest = Path(dest)

    cache_dir = Path(kagglehub.dataset_download(settings.dataset))
    csv_files = sorted(cache_dir.glob(_CSV_GLOB))
    if len(csv_files) != 1:
        raise RuntimeError(
            f"expected exactly one CSV in {settings.dataset!r}, found {len(csv_files)}: "
            f"{[f.name for f in csv_files]}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_files[0], dest)
    return dest


def extract_single_csv(zip_path: Path | str, dest: Path | str) -> Path:
    """Extracts the one CSV inside ``zip_path`` to ``dest``.

    For a zip downloaded manually from Kaggle's website (not via ``kagglehub``,
    which already extracts -- see this module's docstring). Raises if the archive
    doesn't contain exactly one CSV.
    """
    zip_path = Path(zip_path)
    dest = Path(dest)

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(
                f"expected exactly one CSV in {zip_path}, found {len(csv_names)}: {csv_names}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(csv_names[0]) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest
