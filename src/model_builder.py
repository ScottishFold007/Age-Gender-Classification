import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_same_padding(input_size, kernel_size, stride):
    """计算'same'填充所需的填充值"""
    if input_size % stride == 0:
        padding = max(kernel_size - stride, 0)
    else:
        padding = max(kernel_size - (input_size % stride), 0)
    return padding // 2, padding - padding // 2

class FeatureLearningBlock(nn.Module):
    def __init__(self, in_channels, reg_value=None):
        super(FeatureLearningBlock, self).__init__()
        # 使用计算的填充值而不是'same'
        self.conv1 = nn.Conv2d(in_channels, 120, kernel_size=9, stride=2, padding=4)  # padding=4 模拟'same'
        self.bn1 = nn.BatchNorm2d(120)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(120, 256, kernel_size=5, stride=1, padding=2)  # padding=2 模拟'same'
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)  # padding=1 模拟'same'
        self.bn3 = nn.BatchNorm2d(384)
        
        # 添加权重正则化
        if reg_value is not None:
            self.weight_decay = reg_value
        else:
            self.weight_decay = 0
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        return x

class TimeAttention(nn.Module):
    def __init__(self, in_channels, reg_value=None):
        super(TimeAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 9), padding=(0, 4))  # 模拟'same'
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))  # 模拟'same'
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))  # 模拟'same'
        self.bn = nn.BatchNorm2d(64)
        
        # 添加权重正则化
        if reg_value is not None:
            self.weight_decay = reg_value
        else:
            self.weight_decay = 0
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        return x

class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, reg_value=None):
        super(FrequencyAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(9, 1), padding=(4, 0))  # 模拟'same'
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))  # 模拟'same'
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))  # 模拟'same'
        self.bn = nn.BatchNorm2d(64)
        
        # 添加权重正则化
        if reg_value is not None:
            self.weight_decay = reg_value
        else:
            self.weight_decay = 0
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        return x

class MultiAttentionModule(nn.Module):
    def __init__(self, in_channels, reg_value=None):
        super(MultiAttentionModule, self).__init__()
        self.time_attention = TimeAttention(in_channels, reg_value)
        self.freq_attention = FrequencyAttention(in_channels, reg_value)
        self.bn = nn.BatchNorm2d(128)  # 64 + 64 = 128
    
    def forward(self, x):
        ta = self.time_attention(x)
        fa = self.freq_attention(x)
        mam = torch.cat([ta, fa], dim=1)
        mam = self.bn(mam)
        return mam

class VoiceClassifier(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=12, reg_value=None):
        super(VoiceClassifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.reg_value = reg_value
        
        # 第一个特征学习块
        self.flb1 = FeatureLearningBlock(input_shape[0], reg_value)
        
        # 多注意力模块
        self.mam = MultiAttentionModule(384, reg_value)
        
        # 第二个特征学习块 (输入通道数为384+128=512)
        self.flb2 = FeatureLearningBlock(512, reg_value)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(384, 80)  # 输入大小取决于FLB2的输出
        self.bn = nn.BatchNorm1d(80)
        self.fc2 = nn.Linear(80, num_classes)
        
    def forward(self, x):
        # 第一个特征学习块
        x1 = self.flb1(x)
        
        # 多注意力模块
        mam_out = self.mam(x1)
        
        # 连接FLB1和MAM的输出
        x = torch.cat([x1, mam_out], dim=1)
        
        # 第二个特征学习块
        x = self.flb2(x)
        
        # 全连接层
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
    
    def summary(self):
        """打印模型结构摘要"""
        print(f"模型输入形状: {self.input_shape}")
        print(f"分类类别数: {self.num_classes}")
        print(f"权重正则化系数: {self.reg_value}")
        print(f"模型结构:\n{self}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
