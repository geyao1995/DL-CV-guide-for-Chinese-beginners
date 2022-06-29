import logging
import sys
import warnings
from logging import FileHandler, Formatter, StreamHandler
from pathlib import Path

import numpy as np
import torch


class Saver:
    def __init__(self, model, save_dir: Path, parents_dir: bool = False,
                 name_log: str = 'saver', mode_log_file: str = 'a'):
        self.save_dir = save_dir
        self.model = model

        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=parents_dir)
            print(f'Make save dir: {save_dir}')

        self.dir_weights = self.make_subdir('weights')
        self.dir_data = self.make_subdir('data')
        self.logger = self.get_logger(name_log, mode_log_file)

    def make_subdir(self, name_dir: str):
        sub_dir = self.save_dir.joinpath(name_dir)
        if sub_dir.exists():
            warnings.warn(f'{sub_dir} already exist, be careful!')
        else:
            sub_dir.mkdir(exist_ok=True, parents=False)

        return sub_dir

    def get_logger(self, name_log, mode_log_file):
        logger = logging.getLogger(name_log)
        logger.setLevel(logging.DEBUG)

        handler_stream = StreamHandler(sys.stdout)
        handler_stream.setLevel(logging.INFO)
        formatter_stream = Formatter("%(asctime)s - %(message)s",
                                     "%Y-%m-%d %H:%M:%S",  # remove milliseconds
                                     )
        handler_stream.setFormatter(formatter_stream)
        logger.addHandler(handler_stream)

        path_log_file = self.save_dir.joinpath('a.log')
        handler_file = FileHandler(filename=path_log_file, mode=mode_log_file)
        handler_file.setLevel(logging.DEBUG)
        # "%(levelname)s - %(asctime)s - %(message)s - %(name)s"
        formatter_file = Formatter("%(asctime)s - %(message)s",
                                   "%Y-%m-%d %H:%M:%S",  # remove milliseconds
                                   )
        handler_file.setFormatter(formatter_file)
        logger.addHandler(handler_file)
        logger.debug(f'{"":-^30}')

        return logger

    def save_data(self, name: str, data: np.ndarray):
        np.save(self.dir_data.joinpath(name), data)

    def save_weights(self, idx_epoch):
        weights_path = self.dir_weights.joinpath(f'epoch_{idx_epoch}.pth')
        torch.save(self.model.state_dict(), weights_path)

    def log_str_to_file(self, s):
        self.logger.debug(s)

    def close_log(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
