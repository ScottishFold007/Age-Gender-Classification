from data_utils import CommonVoiceDatasetPreprocessor, SpectrogramGenerator, VoiceDataset
from argparse import ArgumentParser
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='Dataset directory')
    parser.add_argument('--dataset_type', default='clean', choices=['raw', 'clean'], help="Dataset type. It can be `raw` or `clean`")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloaders')
    parser.add_argument('--cls_task', default='gender_age', choices=['gender', 'age', 'gender_age'], 
                        help='Classification task: gender, age, or gender_age')
    args = parser.parse_args()
    return args


def get_dataloaders(dataset_dir, batch_size=32, cls_task='gender_age'):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = VoiceDataset(dataset_dir, 'train', cls_task, transform)
    val_dataset = VoiceDataset(dataset_dir, 'dev', cls_task, transform)
    test_dataset = VoiceDataset(dataset_dir, 'test', cls_task, transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"分类任务: {cls_task}, 类别数: {train_dataset.get_num_classes()}")
    print(f"类别名称: {train_dataset.get_class_names()}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parse_args()
    
    # 1. 预处理数据集（如果是原始数据集）
    if args.dataset_type == 'raw':
        print("开始预处理原始数据集...")
        dataset_preprocessor = CommonVoiceDatasetPreprocessor(args.dataset_dir)
        dataset_preprocessor.startPreprocessing()
    
    # 2. 生成频谱图
    print("开始生成频谱图...")
    spectrogram_generator = SpectrogramGenerator(args.dataset_dir)
    spectrogram_generator.startExtractingSpectrograms()
    
    # 3. 创建PyTorch数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        cls_task=args.cls_task
    )
    
    # 4. 显示数据样例
    print("\n数据样例:")
    images, labels = next(iter(train_loader))
    print(f"批次形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    
    print("\n处理完成! 数据已准备好用于训练模型。")
