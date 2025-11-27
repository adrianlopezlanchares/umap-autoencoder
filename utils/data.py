import subprocess
from pathlib import Path
import zipfile


def download_data():
    """
    Download the CelebA dataset from Kaggle to the project root, under /data.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "celeba-dataset.zip"

    subprocess.run(
        [
            "curl",
            "-L",
            "-o",
            str(output_file),
            "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset",
        ],
        check=True,
    )

    with zipfile.ZipFile(output_file, "r") as z:
        z.extractall(data_dir)

    output_file.unlink()
