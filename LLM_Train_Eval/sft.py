import argparse
import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter
from data import *
from model import *
from torch import nn
from data import SftDataset


def parse_arguments():  # 定义一个函数用于解析命令行参数
    parser = argparse.ArgumentParser(description="skyer pretrain")  # 创建一个ArgumentParser对象
    parser.add_argument('--local_rank', type=int, default=-1)  # 添加一个命令行参数--local_rank
    parser.add_argument('--data_file', type=str)  # 添加一个命令行参数--data_file
    parser.add_argument('--ss', type=int)  # 添加一个命令行参数--ss
    parser = deepspeed.add_config_arguments(parser)  # 为DeepSpeed添加配置参数
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回解析后的参数


class Trainer:

    def __init__(self):
        #初始化分布式训练环境
        deepspeed.init_distributed()
        self.args = parse_arguments()
        #获取当前进程
        _rank = deepspeed.comm.get_rank()
        if _rank == 0:
            self.log = SummaryWriter("runs")

        self.model = Skyer()

        # 使用DeepSpeed初始化模型、优化器、数据加载器和学习率调度器
        self.engine, self.opt, self.training_dataloader, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            training_data=SftDataset(f"{self.args.data_file}", 128),
            
            model_parameters=self.model.parameters(),
            #deepspeed配置文件路径
            config="./deepspeed_config.json"
        )
        # 定义损失函数为交叉熵损失，忽略index为0的类别
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):

        _rank = deepspeed.comm.get_rank()

        self.engine.train()
        #加载检查点
        _, client_sd = self.engine.load_checkpoint(f"sft_save")
        if client_sd is None:
            client_sd = {"step": 0}

        for _i, (_prompt,_tag) in enumerate(self.training_dataloader):
            _xs=_prompt[:,:-1].to(device=self.engine.device)
            _ys=_tag[:,1:].to(device=self.engine.device)
            _os = self.engine(_xs)
            # print("_ds",_ds.shape)
            # print("_xs",_xs.shape)
            # print("_ys",_ys.shape)
            # print("_os",_os.shape)
            _os = _os.reshape(-1, 30000)
            _os = _os - _os.mean(-1, keepdim=True)

            _ys = _ys.reshape(-1)
            # print("_ys1", _ys.shape)
            # print("_os1", _os.shape)
            # exit()
            _loss = self.loss_fn(_os, _ys)

            self.engine.backward(_loss)
            self.engine.step()

            _step = client_sd['step']
            if _rank == 0 and _i % 100 == 0:
                #记录损失
                self.log.add_scalar(f"loss", _loss, _step)
            client_sd['step'] += 1

        # hour = datetime.now().hour
        ss = self.args.ss
        # 保存检查点
        self.engine.save_checkpoint(f"sft_save", tag=f"sft_{ss}",
                                    client_state={"step": client_sd['step']})


if __name__ == '__main__':
    train = Trainer()
    train()
