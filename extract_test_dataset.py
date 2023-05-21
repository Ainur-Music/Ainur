import os

def fast_scandir(path: str, exts, recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files

# Adapted from ...
def get_wav_filenames(paths, recursive: bool):
    extensions = [".wav", ".flac"]
    filenames = []
    for path in paths:
        _, files = fast_scandir(path, extensions, recursive=recursive)
        filenames.extend(files)
    return filenames


if __name__ == "__main__":
    import torch
    from torch.utils.data import random_split
    from lightning.pytorch import seed_everything
    seed_everything(42, workers=True)


    dataset = get_wav_filenames(["/Users/gio/spotdl"], recursive=False)

    *_, test_dataset = random_split(dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))

    for x in test_dataset:
        print(x)