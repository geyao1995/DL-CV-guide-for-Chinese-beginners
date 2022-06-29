import numpy as np
import torch.optim as optim

from car_data_loader import get_loader
from config import CarBrandClsConfig
from help_funs import set_seed, get_device
from model import CarBrandClsNet
from saver import Saver
from tester import CarTester
from trainer import CarTrainer


def main():
    set_seed()
    device = get_device()

    model = CarBrandClsNet()
    model.to(device)

    train_loader, test_loader = get_loader()

    optimizer = optim.SGD(model.parameters(), lr=CarBrandClsConfig.lr_max)

    trainer = CarTrainer(device, model, train_loader, optimizer)
    tester = CarTester(device, model, test_loader)
    saver = Saver(model, CarBrandClsConfig.dir_save, parents_dir=True)

    acc_all = []

    for idx_epoch in range(1, CarBrandClsConfig.epoch_total + 1):
        trainer.train_epoch(idx_epoch)
        if idx_epoch % 5 == 0:
            acc = tester.evaluate()
            acc_all.append(acc)
            saver.log_str_to_file(f'Epoch {idx_epoch}, Accuracy:{acc:.2%}')
            saver.save_weights(idx_epoch)

    saver.save_data('acc', np.array(acc_all))
    saver.close_log()


if __name__ == '__main__':
    main()
