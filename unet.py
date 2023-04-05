import torch.nn as nn
import torch


# Leaky relu function with slope 0.2
def lrelu(x):
    outt = torch.max(0.2*x, x)
    return outt

# Original Unet class
class UNet_original(nn.Module):
    def __init__(self):
        super(UNet_original, self).__init__() # Call the init function of the parent class
        # Double convolutional layer and one maxpooling layer
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # One upsample layer, double convolutional layer 
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # One upsample layer, double convolutional layer 
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # One upsample layer, double convolutional layer 
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # One upsample layer, double convolutional layer w
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # One convolutional layer
        self.conv10 = nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv1 = lrelu(self.conv1(x))
        conv1 = lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv2 = lrelu(self.conv2(pool1))
        conv2 = lrelu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv3 = lrelu(self.conv3(pool2))
        conv3 = lrelu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv4 = lrelu(self.conv4(pool3))
        conv4 = lrelu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        # Forward pass through the double convolutional layer leaky relu activation
        conv5 = lrelu(self.conv5(pool4))
        conv5 = lrelu(self.conv5_2(conv5))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up6 = torch.cat([self.up6(conv5), conv4], 1)
        conv6 = lrelu(self.conv6(up6))
        conv6 = lrelu(self.conv6_2(conv6))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up7 = torch.cat([self.up7(conv6), conv3], 1)
        conv7 = lrelu(self.conv7(up7))
        conv7 = lrelu(self.conv7_2(conv7))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up8 = torch.cat([self.up8(conv7), conv2], 1)
        conv8 = lrelu(self.conv8(up8))
        conv8 = lrelu(self.conv8_2(conv8))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up9 = torch.cat([self.up9(conv8), conv1], 1)
        conv9 = lrelu(self.conv9(up9))
        conv9 = lrelu(self.conv9_2(conv9))

        # Forward pass through the convolutional layer
        conv10 = self.conv10(conv9)

        # Pixel shuffle layer
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


# Unet class with a single batch normalization layer
class UNet_single_batchnorm(nn.Module):
    def __init__(self):
        super(UNet_single_batchnorm, self).__init__() # Call the init function of the parent class
        # Double convolutional layer and one maxpooling layer
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # One upsample layer, double convolutional layer 
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # One upsample layer, double convolutional layer 
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # One upsample layer, double convolutional layer 
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(64)

        # One upsample layer, double convolutional layer w
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)

        # One convolutional layer
        self.conv10 = nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv1 = lrelu(self.conv1(x))
        conv1 = lrelu(self.bn1(self.conv1_2(conv1)))
        pool1 = self.pool1(conv1)
        
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv2 = lrelu(self.conv2(pool1))
        conv2 = lrelu(self.bn2(self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv3 = lrelu(self.conv3(pool2))
        conv3 = lrelu(self.bn3(self.conv3_2(conv3)))
        pool3 = self.pool3(conv3)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv4 = lrelu(self.conv4(pool3))
        conv4 = lrelu(self.bn4(self.conv4_2(conv4)))
        pool4 = self.pool4(conv4)

        # Forward pass through the double convolutional layer leaky relu activation
        conv5 = lrelu(self.conv5(pool4))
        conv5 = lrelu(self.bn5(self.conv5_2(conv5)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up6 = torch.cat([self.up6(conv5), conv4], 1)
        conv6 = lrelu(self.conv6(up6))
        conv6 = lrelu(self.bn6(self.conv6_2(conv6)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up7 = torch.cat([self.up7(conv6), conv3], 1)
        conv7 = lrelu(self.conv7(up7))
        conv7 = lrelu(self.bn7(self.conv7_2(conv7)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up8 = torch.cat([self.up8(conv7), conv2], 1)
        conv8 = lrelu(self.conv8(up8))
        conv8 = lrelu(self.bn8(self.conv8_2(conv8)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up9 = torch.cat([self.up9(conv8), conv1], 1)
        conv9 = lrelu(self.conv9(up9))
        conv9 = lrelu(self.bn9(self.conv9_2(conv9)))

        # Forward pass through the convolutional layer
        conv10 = self.conv10(conv9)

        # Pixel shuffle layer
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


# Unet class with double normalization layer
class UNet_double_batchnorm(nn.Module):
    def __init__(self):
        super(UNet_double_batchnorm, self).__init__() # Call the init function of the parent class
        # Double convolutional layer and one maxpooling layer
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Double convolutional layer and one maxpooling layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)

        # One upsample layer, double convolutional layer 
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)

        # One upsample layer, double convolutional layer 
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(128)

        # One upsample layer, double convolutional layer 
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(64)

        # One upsample layer, double convolutional layer w
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn9_1 = nn.BatchNorm2d(32)

        # One convolutional layer
        self.conv10 = nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv1 = lrelu(self.bn1(self.conv1(x)))
        conv1 = lrelu(self.bn1_1(self.conv1_2(conv1)))
        pool1 = self.pool1(conv1)
        
        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv2 = lrelu(self.bn2(self.conv2(pool1)))
        conv2 = lrelu(self.bn2_1(self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv3 = lrelu(self.bn3(self.conv3(pool2)))
        conv3 = lrelu(self.bn3_1(self.conv3_2(conv3)))
        pool3 = self.pool3(conv3)

        # Forward pass through the double convolutional layer leaky relu activation and maxpooling layer
        conv4 = lrelu(self.bn4(self.conv4(pool3)))
        conv4 = lrelu(self.bn4_1(self.conv4_2(conv4)))
        pool4 = self.pool4(conv4)

        # Forward pass through the double convolutional layer leaky relu activation
        conv5 = lrelu(self.bn5(self.conv5(pool4)))
        conv5 = lrelu(self.bn5_1(self.conv5_2(conv5)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up6 = torch.cat([self.up6(conv5), conv4], 1)
        conv6 = lrelu(self.bn6(self.conv6(up6)))
        conv6 = lrelu(self.bn6_1(self.conv6_2(conv6)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up7 = torch.cat([self.up7(conv6), conv3], 1)
        conv7 = lrelu(self.bn7(self.conv7(up7)))
        conv7 = lrelu(self.bn7_1(self.conv7_2(conv7)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up8 = torch.cat([self.up8(conv7), conv2], 1)
        conv8 = lrelu(self.bn8(self.conv8(up8)))
        conv8 = lrelu(self.bn8_1(self.conv8_2(conv8)))

        # Forward pass through the upsample layer and double convolutional layer with leaky relu activation
        up9 = torch.cat([self.up9(conv8), conv1], 1)
        conv9 = lrelu(self.bn9(self.conv9(up9)))
        conv9 = lrelu(self.bn9_1(self.conv9_2(conv9)))

        # Forward pass through the convolutional layer
        conv10 = self.conv10(conv9)

        # Pixel shuffle layer
        out = nn.functional.pixel_shuffle(conv10, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)