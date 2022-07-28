import sys

import torch
from tqdm import tqdm


class CarTester:
    def __init__(self, device, model, loader_test):
        self.model = model
        self.loader_test = loader_test
        self.device = device

    def evaluate(self):
        self.model.eval()
        
        correct = 0
        total = 0

        iterator_tqdm = tqdm(self.loader_test, file=sys.stdout, position=0, ncols=100)

        with torch.no_grad():
            for i, batch in enumerate(iterator_tqdm):
                inputs = batch[0].to(self.device, dtype=torch.float)
                labels = batch[1].to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterator_tqdm.set_description_str(f'Test on {total} examples. '
                                                  f'Accuracy-{correct / total:.2%}')
        iterator_tqdm.close()

        return correct / total
