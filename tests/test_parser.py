import pathlib

from training import parser

RUN_DIR = "/home/alberto/Data/master/TFM/colab/tensorboard/dbp_1c/s1"

if __name__ == "__main__":
    last = parser.extract_latest(RUN_DIR, "s")
    print(pathlib.Path(last).name)

    log = parser.extract_log(RUN_DIR)
    print("latest log:\n", log)

