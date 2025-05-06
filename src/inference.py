from model_utils import AgeGenderDetector
from argparse import ArgumentParser
import logging
import os

logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path', default='models/', help='模型路径')
    parser.add_argument('--audio_file', default='data/CommonVoice/Audio/cv-valid-test/sample-000001.mp3', help='音频文件路径')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="分类任务")
    args = parser.parse_args()
    return args

def detect(args):
    # 检查文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件不存在: {args.audio_file}")
        return
    
    # 如果提供的是目录而不是文件，则自动查找对应任务的最佳模型
    model_path = args.model_path
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, args.cls_task, f'best_model_{args.cls_task}.pth')
        if not os.path.exists(model_path):
            print(f"错误: 在指定目录中找不到模型文件: {model_path}")
            return
    
    # 评估最佳模型
    print(f"使用模型: {model_path}")
    print(f"分析音频文件: {args.audio_file}")
    age_gender_detector = AgeGenderDetector(model_path, args.cls_task)
    detection = age_gender_detector.getPrediction(args.audio_file)
    print('模型预测 ==> ', detection)

if __name__ == '__main__':
    args = parse_args()
    detect(args)
