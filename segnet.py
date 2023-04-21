import torch
import torch.nn as nn

class NConv(nn.Module):
    def __init__(self, in_channels, out_channels, n):
        super(NConv, self).__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        # Check what is the value of padding and stride and filter size in VGG16
        self.conv.append(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1))
        for i in range(n-1):
            self.conv.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            self.conv.append(nn.BatchNorm2d(out_channels))
            self.conv.append(nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n, k_size):
        super(EncoderLayer, self).__init__()
        self.eLayer = nn.Sequential(
            NConv(in_channels, out_channels, n),
            nn.MaxPool2d(kernel_size=k_size, return_indices=True)
        )
    '''
    forward returns output and indices from the MaxPool2d layer
    '''

    def forward(self, x):
        return self.eLayer(x)


class Encoder(nn.Module):
    def __init__(self, in_channels ,features=[[64, 2, 2], [128, 2, 2], [256, 3, 2], [512, 3, 2], [512, 3, 2]]):
        '''
        Format = [channels, conv layers ( for NConve ), kernel_size ( for MaxPool )]
        '''
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        self.indices = []
        self.sizes = []
        for feature in features:
            out_channels = feature[0]
            numLayers = feature[1]
            k_size = feature[2]
            self.encoder.append(EncoderLayer(in_channels, out_channels, numLayers, k_size))
            in_channels = out_channels
    
    def forward(self, x):
        for layer in self.encoder:
            self.sizes.append(x.size())
            x, indicex = layer(x)
            self.indices.append(indicex)
        return x, self.indices, self.sizes 


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n, k_size):
        super(DecoderLayer, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=k_size),
        self.fcn = NConv(in_channels, out_channels, n)
        

    def forward(self, x, indices, op_size):
        out = self.unpool(x, indices, output_size=op_size)
        out = self.fcn(x)
        return out


class Decoder(nn.Module):
    def __init__(self, features=[[64, 2, 2], [128, 2, 2], [256, 3, 2], [512, 3, 2], [512, 3, 2]], k=2):
        super(Decoder, self).__init__()
        features.reverse()
        self.decoder = nn.Sequential(); 
        n = len(features)
        for i in range(n-1):
            in_channels = features[0]
            out_channels = features[0]
            numConvLayers = features[1]
            k_size = features[2]
            self.decoder.append(DecoderLayer(in_channels, out_channels, numConvLayers, k_size))
        self.decoder.append(NConv(features[n-1][0], k, 1))
    
    def forward(self, x, indices, size):
        indices.reverse()
        size.reverse()
        for i, layer in enumerate(self.decoder):
            if i==len(self.decoder)-1:
                break
            x = layer(x, indices[i], size[i])
        x = self.decoder[len(self.decoder)-1](x)    # final conv for 64 to k - classes
        return x


class SegNet(nn.Module):
    def __init__(self, in_channels=3, features=[[64, 2, 2], [128, 2, 2], [256, 3, 2], [512, 3, 2], [512, 3, 2]]):
        super(SegNet, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features)
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x, indices, sizes = self.encoder(x)
        x = self.decoder(x, indices, sizes)
        x = self.sigmoid(x)

        return x
    
# def test():
#     x = torch.randn((3, 3, 240, 320))
#     model = SegNet(in_channels=3)
#     predictions = model(x)
#     assert predictions.shape == x.shape

# if __name__=="__main__":
#     test()

        









