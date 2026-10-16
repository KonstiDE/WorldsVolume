import math
import os
import glob
import random
import shutil

import config.config as cfg


def move_into_split():
    splits = cfg.get_split_map()

    files_list = glob.glob(os.path.join(cfg.get_frame_path(), "*.tif"))
    tpv = len(files_list)

    random.seed(25111999)
    random.shuffle(files_list)

    train_percent = math.floor(tpv * splits["train"][0])
    validation_percent = math.floor(tpv * splits["validation"][0])
    test_percent = tpv - train_percent - validation_percent


    for i in range(train_percent):
        file = files_list[i]
        shutil.move(
            file, os.path.join(cfg.get_frame_path(), "train", os.path.basename(str(file)))
        )
    for i in range(validation_percent):
        file = files_list[i + train_percent]
        shutil.move(
            file, os.path.join(cfg.get_frame_path(), "validation", os.path.basename(str(file)))
        )
    for i in range(test_percent):
        file = files_list[i + train_percent + validation_percent]
        shutil.move(
            file, os.path.join(cfg.get_frame_path(), "test", os.path.basename(str(file)))
        )



if __name__ == '__main__':
    move_into_split()

