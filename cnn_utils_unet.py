import torch
import torch.nn as nn
from physics_utils import PhysicalLayer


# AG This code was taken from
# ://debuggercafe.com/unet-from-scratch-using-pytorch/

def double_convolution_2d(in_channels, out_channels, dropout=0.0):
    """
    Convolution block with optional dropout.
    """
    conv_layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    ]
    return nn.Sequential(*conv_layers)

def double_convolution_3d(in_channels, out_channels, dropout=0.0):
    """
    Convolution block with optional dropout.
    """
    kernel_size = (2, 3, 3)
    padding = 1
    conv_layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm3d(out_channels),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm3d(out_channels)
    ]
    return nn.Sequential(*conv_layers)


class OpticsDesignUnet(nn.Module):
    def __init__(self, config):
        super(OpticsDesignUnet, self).__init__()
        # adding the physicalLayer into the mix
        self.Nimgs = config['Nimgs']
        self.physicalLayer = PhysicalLayer(config)
        self.conv3d = config.get('conv3d', False)  # flag for 3D convolutions
        
        num_classes = config['num_classes']
        dropout = config.get('dropout', 0.0)  # dropout value from config

        # The code from //debuggercafe.com/unet-from-scratch-using-pytorch/
        

        # Contracting path.
        # Each convolution is applied twice.
        # AG since we use monochrome here, we used only one channel as an input
        # For four layers we had ~14M parameters, too many
        # For Three layers we hopefully have few parameters to optimize
        if self.conv3d == False:
            self.norm = nn.BatchNorm2d(num_features=self.Nimgs, affine=True)
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.down_convolution_1 = double_convolution_2d(self.Nimgs, 64, dropout=dropout)
            self.down_convolution_2 = double_convolution_2d(64, 128, dropout=dropout)
            self.down_convolution_3 = double_convolution_2d(128, 256, dropout=dropout)
            
            # Expanding path.
            self.up_transpose_1 = nn.ConvTranspose2d(
                in_channels=256, out_channels=128,
                kernel_size=2,
                stride=2)
            self.up_convolution_1 = double_convolution_2d(256, 128, dropout=dropout)

            self.up_transpose_2 = nn.ConvTranspose2d(
                in_channels=128, out_channels=64,
                kernel_size=2,
                stride=2)
            self.up_convolution_2 = double_convolution_2d(128, 64, dropout=dropout)

            # output => `out_channels` as per the number of classes.
            self.out = nn.Conv2d(
                in_channels=64, out_channels=num_classes,
                kernel_size=1
            )
            
            
        else:
            self.norm = nn.BatchNorm3d(num_features=1, affine=True)
            self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.down_convolution_1 = double_convolution_3d(1, 64, dropout=dropout)
            self.down_convolution_2 = double_convolution_3d(64, 128, dropout=dropout)
            self.down_convolution_3 = double_convolution_3d(128, 256, dropout=dropout)
            
            # Expanding path.
            self.up_transpose_1 = nn.ConvTranspose3d(
                in_channels=256, out_channels=128,
                kernel_size=2,
                stride=2)
            self.up_convolution_1 = double_convolution_3d(256, 128, dropout=dropout)

            self.up_transpose_2 = nn.ConvTranspose3d(
                in_channels=128, out_channels=64,
                kernel_size=2,
                stride=2)
            self.up_convolution_2 = double_convolution_3d(128, 64, dropout=dropout)

            # output => `out_channels` as per the number of classes.
            self.out = nn.Conv3d(
                in_channels=64, out_channels=num_classes,
                kernel_size=1
            )
            

        

    def forward(self, mask, xyz, Nphotons):
        im = self.physicalLayer(mask, xyz, Nphotons)
        im = self.norm(im)

        # im_test = im[0,0,:,:]
        # plt.imshow(im_test.detach().numpy())
        # plt.show()
        down_1 = self.down_convolution_1(im)
        down_2 = self.max_pool(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool(down_3)
        down_5 = self.down_convolution_3(down_4)

        up_1 = self.up_transpose_1(down_5)
        im = self.up_convolution_1(torch.cat([down_3, up_1], 1))
        up_2 = self.up_transpose_2(im)
        im = self.up_convolution_2(torch.cat([down_1, up_2], 1))
        out = self.out(im)
        return out

if __name__ == '__main__':
    input_image = torch.rand((1, 3, 512, 512))
    config = {'num_classes' : 1}
    model = OpticsDesignUnet(config)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)
    print(outputs.shape)