import torch
import torch.nn as nn
import torch.nn.functional as F
from BRA import BiLevelRoutingAttention_nchw
from SCConv import ScConv
from MHSA import MHSA


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate_feature_map(feature_map, target_size):
    # feature_map = feature_map.unsqueeze(0)
    return torch.nn.functional.interpolate(feature_map, size=target_size, mode='bilinear', align_corners=False)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DBANet(nn.Module):
    def __init__(self, in_channels, img_size, num_channels=[64, 32, 16]):
        super(DBANet, self).__init__()
        self.in_channels = in_channels
        self.img_szie = img_size
        num_channels_1, num_channels_2, num_channels_3 = num_channels

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.act_function = nn.GELU()

        self.LN1 = nn.LayerNorm(normalized_shape=[3, 128, 128])

        self.conv_layer1 = nn.Conv2d(in_channels=self.in_channels, out_channels=num_channels_1, kernel_size=5, stride=2,
                                     padding=2)
        self.conv_layer2 = nn.Conv2d(in_channels=self.in_channels, out_channels=num_channels_2, kernel_size=3, stride=2,
                                     padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=self.in_channels, out_channels=num_channels_3, kernel_size=3, stride=1,
                                     padding=1)

        self.conv_layer1_test = nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_2, kernel_size=3,
                                          stride=2, padding=1)
        self.conv_layer3_test = nn.Conv2d(in_channels=num_channels_2, out_channels=num_channels_3, kernel_size=3,
                                          stride=1, padding=1)

        self.up_pixel_shuffle = nn.PixelShuffle(2)
        self.up_conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=4, padding=1)

        self.SCConv_model_up = ScConv(32)
        self.SCConv_model_low = ScConv(16)

        self.model_MHSA = MHSA(n_dims=32)

        self.block_BRA = BiLevelRoutingAttention_nchw(16)

        self.conv_after_BRA = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_test_converto128 = nn.Conv2d(32, 3, kernel_size=1)

        self.conv_before_HRM = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_before_hrm = nn.ReLU(inplace=True)

        self.conv_test1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        self.pix_test1 = nn.PixelShuffle(2)
        self.conv_test2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)

        self.conv_layer_last = nn.Sequential(nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                                   nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        original = x

        x = self.LN1(x)

        output1 = self.conv_layer1(x)
        output1 = self.act_function(output1)

        output2 = self.conv_layer2(x)
        output2 = self.act_function(output2)

        output1_test = self.conv_layer1_test(output1)

        output3 = self.conv_layer3(x)
        output3 = self.act_function(output3)

        output2_test = self.conv_layer3_test(output2)

        expanded_feature_map1 = interpolate_feature_map(output1_test, 64)
        X_up = output2 + expanded_feature_map1
        expanded_feature_map2 = interpolate_feature_map(output2_test, 128)
        X_low = output3 + expanded_feature_map2

        SCConv_model_up = self.SCConv_model_up(X_up)
        SCConv_model_low = self.SCConv_model_low(X_low)

        model_MHSA_up = self.model_MHSA(SCConv_model_up)

        block_BRA_low = self.block_BRA(SCConv_model_low)
        block_BRA_low = self.block_BRA(block_BRA_low)
        block_BRA_low = self.block_BRA(block_BRA_low)

        y_uppix = self.up_pixel_shuffle(block_BRA_low)
        X_uppix_low = self.up_conv1(y_uppix)

        x = model_MHSA_up + X_uppix_low

        x = self.conv_test_converto128(x)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        x = x + original

        x = self.conv_before_HRM(x)
        x = self.relu_before_hrm(x)

        x = self.conv_test1(x)

        x = self.pix_test1(x)

        x = self.conv_test2(x)
        x = self.pix_test1(x)

        x = self.conv_layer_last(x)

        # x = self.upsample(x)

        return x



