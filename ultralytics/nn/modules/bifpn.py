import torch
import torch.nn as nn
from .conv import Conv 

class BiFPN_Add2(nn.Module):
    """
    BiFPN untuk 2 input dengan Auto-Projection.
    """
    def __init__(self, c1, c2):
        super().__init__()
        # c1 adalah list channel input, contoh: [512, 256]
        # c2 adalah channel output yang diinginkan, contoh: 256
        
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        
        # Buat proyeksi otomatis untuk setiap input
        self.input_convs = nn.ModuleList()
        for in_channels in c1:
            if in_channels != c2:
                # Jika channel beda, lakukan Conv 1x1 untuk menyamakan
                self.input_convs.append(Conv(in_channels, c2, 1, 1))
            else:
                # Jika channel sudah sama, biarkan lewat (Identity)
                self.input_convs.append(nn.Identity())

    def forward(self, x):
        # x adalah list tensor [input1, input2]
        
        # 1. Samakan channel semua input
        inputs = [conv(t) for conv, t in zip(self.input_convs, x)]
        
        # 2. Hitung bobot normalisasi
        w = self.w.relu() / (self.w.relu().sum() + self.epsilon)
        
        # 3. Weighted Sum
        return w[0] * inputs[0] + w[1] * inputs[1]

class BiFPN_Add3(nn.Module):
    """
    BiFPN untuk 3 input dengan Auto-Projection.
    """
    def __init__(self, c1, c2):
        super().__init__()
        # c1 adalah list channel input, contoh: [256, 512, 256]
        
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        
        self.input_convs = nn.ModuleList()
        for in_channels in c1:
            if in_channels != c2:
                self.input_convs.append(Conv(in_channels, c2, 1, 1))
            else:
                self.input_convs.append(nn.Identity())

    def forward(self, x):
        inputs = [conv(t) for conv, t in zip(self.input_convs, x)]
        w = self.w.relu() / (self.w.relu().sum() + self.epsilon)
        return w[0] * inputs[0] + w[1] * inputs[1] + w[2] * inputs[2]