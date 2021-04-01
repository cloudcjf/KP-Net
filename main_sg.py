#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import tab_printer
# from sg_net import SGTrainer
from sg_net_new import SGTrainer
from parser_sg import sgpr_args
import sys

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    # 传入配置参数,启动程序的时候可以指定路径,否则就调用默认路径里的yml文件
    args = sgpr_args()
    if len(sys.argv)>1:
        args.load(sys.argv[1])
    else:
        args.load('./config/config.yml')
    # 以表格形式打印出传入的参数值
    tab_printer(args)
    # 传入参数给SGnet
    #  sg_net.py --> dgcnn.py --> layers_batch.py
    trainer = SGTrainer(args,True)
    # 模型训练,拟合
    trainer.fit()
    # 在测试数据上验证
    trainer.score()
    
if __name__ == "__main__":
    main()


# bash command

# python main_sg.py ./config/config.yml