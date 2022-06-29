from dataclasses import dataclass

from pathlib import Path


@dataclass
class CarBrandClsConfig:  # config for car brand classification task
    root_images = Path('your_data_path')
    dir_save = Path('results')

    epoch_total = 20
    lr_max = 0.01
    size_train_batch = 20
    size_test_batch = 12

    size_input = (256, 256)
