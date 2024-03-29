

import shutil


def read_doc(file, filepath: str) -> str:
    """Read the document bytestream and return the copied file location"""
    try:
        with open(
                filepath,
                "wb",
        ) as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as exc:
        return (str(exc))
    finally:
        buffer.close()

    return filepath