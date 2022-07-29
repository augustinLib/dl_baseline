from numpy import indices
import torch
from zmq import device

def load_mnist(is_train = True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def split_data(x, y, train_ratio = .8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    indices = torch.randperm(x.size(0))

    x = torch.index_select(
        x, dim=0, index=indices
    ).split([train_cnt, valid_cnt])

    y = torch.index_select(
        y, dim=0, index=indices
    ).split([train_cnt, valid_cnt])

    return x, y

# 등차수열로 줄어드는 hidden_layer size 생성
def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers -1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
