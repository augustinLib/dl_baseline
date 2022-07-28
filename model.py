import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self,
                input_size,
                output_size,
                use_batch_norm = True,
                dropout_p = .4,):

        self.input_size = input_size
        self,output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        y = self.block(x)
        return y

class Classifier(nn.Module):

    def __init__(self,
                input_size,
                output_size,
                hidden_sizes = [500, 400, 300, 200, 100],
                use_batch_norm = True,
                dropout_p = .3):

        super().__init__()

        #hidden_size 설정되지 않았을 때
        assert len(hidden_sizes) > 0, "Need to specify hidden layers size"

        # 첫 hidden layer size input size로 설정
        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            # last_hidden_size 갱신
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            # blocks 해제
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim = -1)
        )
    
    def forward(self, x):
        y = self.layers(x)
        return y
        

