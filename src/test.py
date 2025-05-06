from pytorch_model_utils import ModelEvaluator
from data_utils import get_dataloaders
from argparse import ArgumentParser
import logging
import os

logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='数据集目录')
    parser.add_argument('--model_path', default='models/', help='模型路径')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="分类任务")
    parser.add_argument('--batch_size', default=128, type=int, help='批次大小')
    parser.add_argument('--feature_type', default='spectrogram', 
                        choices=['spectrogram', 'melspectrogram', 'mfcc'], 
                        help='特征类型：spectrogram（频谱图）, melspectrogram（梅尔频谱图）, 或 mfcc（梅尔频率倒谱系数）')
    parser.add_argument('--img_size', type=int, default=64, help='图像大小')
    args = parser.parse_args()
    return args

def test_model(args):
    # 如果提供的是目录而不是文件，则自动查找对应任务的最佳模型
    model_path = args.model_path
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, args.cls_task, f'best_model_{args.cls_task}.pth')
        if not os.path.exists(model_path):
            print(f"错误: 在指定目录中找不到模型文件: {model_path}")
            return
    
    print(f"加载数据集，特征类型: {args.feature_type}")
    # 只需要测试集数据加载器
    _, _, test_loader = get_dataloaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        cls_task=args.cls_task,
        feature_type=args.feature_type,
        img_size=args.img_size
    )
    
    # 评估最佳模型
    print(f"使用模型: {model_path}")
    model_evaluator = ModelEvaluator(args.dataset_dir, args.cls_task, model_path)
    model_evaluator.evaluateBestModel(test_loader)

if __name__ == '__main__':
    args = parse_args()
    test_model(args)
