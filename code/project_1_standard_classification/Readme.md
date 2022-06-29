# 项目一：图像分类

这是供参考的项目代码。

## 任务说明

数据由三种品牌（Benz, BMW, Audi）的汽车照片组成，已经划分好训练集和测试集。训练集中，每个品牌/类别有20张训练样本，4张测试样本。你需要训练一个 CNN 模型，对测试图片进行分类。

由于样本较少，不要求太高的测试准确率，达到50%即可。但注意以下几点：

1. 整个项目的架构是否清晰？（将代码拆分为多个易读的文件）
2. 是否具有一定的日志信息？（可以借助 [tqdm](https://tqdm.github.io/) 模块和 [logging](https://docs.python.org/zh-cn/3/library/logging.html#module-logging) 模块）
3. 如何加载图片数据？（用两种方式实现，1-继承 [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 类；2-继承 [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#imagefolder) 类）
4. 如何管理超参数？（可以借助 [dataclasses](https://docs.python.org/zh-cn/3/library/dataclasses.html#module-dataclasses) 模块）

## 需要额外安装的 Python 包

如果你已经安装好 PyTorch，还需要额外安装以下两个 Python 包

- tqdm
- Matplotlib

在本项目中，tqdm 用来显示程序运行进度，Matplotlib 用来观察一系列变换后的图片。

## 项目文件说明

- `car_data_loader.py`: 加载图片进行训练或测试，直接运行可以观察变换后的图片。
- `config.py`: 存放超参数和数据集/结果保存的路径。
- `help_funs.py`: 一些实用的函数。
- `model.py`: 定义网络模型的结构。
- `saver.py`: 定义 `Saver` 类，用来进行数据的保存和记录。
- `tester.py`: 定义 `Tester` 类，用来测试模型的 checkpoint。
- `train.py`: 主文件，运行它开始训练和测试过程。
- `trainer.py`: 定义 `Trainer` 类，用来训练模型。