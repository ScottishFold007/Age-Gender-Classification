from pytorch_model_utils import ModelTrainer, ModelEvaluator
from data_utils import get_dataloaders
from argparse import ArgumentParser
import logging
import os

logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='数据集目录')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="分类任务")
    parser.add_argument('--batch_size', default=128, type=int, help='批次大小')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='学习率')
    parser.add_argument('--num_epochs', default=50, type=int, help='训练轮数')
    parser.add_argument('--show_summary', action='store_true', help='显示模型摘要')
    parser.add_argument('--show_live_plot', action='store_true', help='显示实时损失和准确率图表')
    parser.add_argument('--feature_type', default='spectrogram', 
                        choices=['spectrogram', 'melspectrogram', 'mfcc'], 
                        help='特征类型：spectrogram（频谱图）, melspectrogram（梅尔频谱图）, 或 mfcc（梅尔频率倒谱系数）')
    parser.add_argument('--img_size', type=int, default=64, help='图像大小')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    args = parser.parse_args()
    return args

def train_model(args):
    # 创建结果和模型目录
    os.makedirs(f'results/{args.cls_task}', exist_ok=True)
    os.makedirs(f'models/{args.cls_task}', exist_ok=True)
    
    # 获取数据加载器
    print(f"加载数据集，特征类型: {args.feature_type}")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_dir, 
        batch_size=args.batch_size, 
        cls_task=args.cls_task,
        feature_type=args.feature_type,
        img_size=args.img_size
    )
    
    # 训练模型
    print(f"开始训练 {args.cls_task} 分类模型")
    model_trainer = ModelTrainer(args.dataset_dir, args.cls_task)
    model_trainer.setTrainingParameters(args.batch_size, args.learning_rate, args.num_epochs)
    model_trainer.startTraining(train_loader, val_loader, args.show_summary, args.show_live_plot)
    
    # 如果需要评估模型
    if args.evaluate:
        best_model_path = model_trainer.getBestModelFilepath()
        print(f"评估最佳模型: {best_model_path}")
        model_evaluator = ModelEvaluator(args.dataset_dir, args.cls_task, best_model_path)
        model_evaluator.evaluateBestModel(test_loader)

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
