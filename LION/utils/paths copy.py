import os
import pathlib

#: The datasets' folders, inside LION_DATA_PATH.
DATA_FOLDERS = {
    "LION_DATA_PATH": "",
    # as downloaded
    "LUNA_DATASET_PATH": "raw/LUNA16",
    "WALNUT_DATASET_PATH": "raw/walnuts",
    "LIDC_IDRI_PATH": "raw/LIDC-IDRI",
    "DETECT_PATH": "raw/2detect",
    # ready for training
    "LUNA_PROCESSED_DATASET_PATH": "processed/LUNA16",
    "LIDC_IDRI_PROCESSED_DATASET_PATH": "processed/LIDC-IDRI",
    "DETECT_PROCESSED_DATASET_PATH": "processed/2detect",
}


def _folder(variable):
    root = os.environ.get(variable)
    if root is None:
        return None
    return pathlib.Path(root).expanduser().resolve()


## Data ready for training use
LUNA_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/LUNA16")
LIDC_IDRI_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/LIDC-IDRI")
DETECT_PROCESSED_DATASET_PATH = LION_DATA_PATH.joinpath("processed/2detect")

# Optional PCM datasets. Override LION_TTV22_PATH on other machines.
ROOT = os.environ.get("LION_STORAGE_ROOT", "/")
TTV22_PATH = (
    pathlib.Path(
        os.environ.get("LION_TTV22_PATH", str(pathlib.Path(ROOT) / "store/LION/ttv22"))
    )
    .expanduser()
    .resolve()
)
PCM_SYNTHETIC_PNG_PATH = (
    TTV22_PATH / "pcm/data/PCM Synthetic Data Set/Synthetic Data/Synthetic Data"
)
PCM_EXAMPLE_DATA_PATH = TTV22_PATH / "pcm/data/photocurrent_data"
PCM_REAL_DATA_EXAMPLE_PATH = TTV22_PATH / "pcm/data/example_data"
