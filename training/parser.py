import glob
import os
import re
import pathlib


def extract_latest(run_dir, prefix="0"):
    run_dir = pathlib.Path(run_dir)
    results_dir = run_dir.parent
    nss = sorted(glob.glob(os.path.join(results_dir, f'{prefix}*', 'network-*.pkl')))

    if nss:
        latest = nss[-1]
    else:
        raise Exception(f"not networks found in {results_dir}")

    return latest


def extract_log(run_dir):
    with open(pathlib.Path(run_dir) / "log.txt") as f:
        lines = [line.strip() for line in f.readlines() if line.startswith("tick")]
    last = lines[-1]

    tick = re.search("tick\s+(\d+)", last).group(1)
    kimg = re.search("kimg\s+(\d+.\d+)", last).group(1)
    augment = re.search("augment\s+(\d+.\d+)", last).group(1)

    return {"tick": int(tick), "kimg": float(kimg), "augment": float(augment)}
