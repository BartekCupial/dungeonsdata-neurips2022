import shutil 
import tempfile

from pathlib import Path


def delete_temp_files():
    directory = tempfile.gettempdir()

    i = 0
    for path in Path(directory).iterdir():
        if path.name.startswith("nle"):
            try:
                shutil.rmtree(path)
                i += 1
            except Exception as e:
                print(f"Error deleting dir: {path}\n{str(e)}")
    print(f"Deleted {i} dirs")
