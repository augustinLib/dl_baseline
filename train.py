import argparse
from pkgutil import get_loader

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Classifier
from trainer import Trainer
from utils import get_hidden_sizes

from loader import *

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()


    return config


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, vaild_loader, test_loader = get_loaders(config)


    print("train:", len(train_loader.dataset))
    print("valid:", len(vaild_loader.dataset))
    print("test:", len(test_loader.dataset))

    input_size = 28*28
    output_size = 10

    model = Classifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm= not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    critic = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(critic)

    trainer = Trainer(config)

    trainer.train(model, critic, optimizer, train_loader, vaild_loader)    
    

if __name__ == '__main__':
    config = define_argparser()
    main(config)
