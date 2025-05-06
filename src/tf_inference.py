'''
python tf_inference.py --model_path age_gender_models/age_classification_acc_81.h5 --cls_task age --audio_file path/to/audio.mp3
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告

class TFAgeGenderDetector:
    def __init__(self, model_path, cls_task):
        """
        初始化TensorFlow模型检测器
        
        参数:
            model_path: TensorFlow模型路径(.h5文件)
            cls_task: 分类任务类型 ('age', 'gender', 或 'gender_age')
        """
        self.cls_task = cls_task
        
        # 加载TensorFlow模型
        print(f"加载模型: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # 初始化标签解码器
        self._decode_gender_age = {0:'female-teens', 1:'female-twenties', 2:'female-thirties', 3:'female-fourties', 4:'female-fifties', 5:'female-sixties',  
                                6:'male-teens', 7:'male-twenties', 8:'male-thirties', 9:'male-fourties', 10:'male-fifties', 11:'male-sixties'}
        self._decode_gender = {0:'male', 1:'female'}
        self._decode_age = {0:'teens', 1:'twenties', 2:'thirties', 3:'fourties', 4:'fifties', 5:'sixties'}
        
        # 设置音频处理参数
        self.sampling_rate = 16000
        self.n_fft = 256
        self.num_overlap = 128
        self._spec_img_size = (64, 64)
    
    def scale_minmax(self, X, min=0, max=255):
        """缩放数组到指定范围"""
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
    
    def saveSpectrogram(self, data, fn):
        """保存频谱图为图像文件"""
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        fig = plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _extractSpectrogram(self, audio_file):
        """从音频文件提取频谱图"""
        y, sr = librosa.load(audio_file, sr=self.sampling_rate)
        spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
        spec = librosa.amplitude_to_db(np.abs(spec))
        # min-max scale to fit inside 8-bit range
        img = self.scale_minmax(spec).astype(np.uint8)
        out_file = 'temp.png'
        self.saveSpectrogram(img, out_file)
        return out_file
    
    def getPredictionLabelName(self, encod_lab):
        """获取预测标签的名称"""
        if self.cls_task == 'age':
            return self._decode_age.get(encod_lab)
        elif self.cls_task == 'gender':
            return self._decode_gender.get(encod_lab)
        elif self.cls_task == 'gender_age':
            return self._decode_gender_age.get(encod_lab)
    
    def getPrediction(self, audio_file):
        """获取音频文件的预测结果"""
        # 提取频谱图
        spec_file = self._extractSpectrogram(audio_file)
        
        # 加载图像
        from tensorflow.keras.utils import load_img, img_to_array
        img = load_img(spec_file, target_size=self._spec_img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 预测
        prediction = self.model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        
        # 删除临时文件
        if os.path.exists(spec_file):
            os.remove(spec_file)
        
        return self.getPredictionLabelName(predicted_label)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, help='TensorFlow模型路径(.h5文件)')
    parser.add_argument('--audio_file', required=True, help='音频文件路径')
    parser.add_argument('--cls_task', required=True, choices=['age', 'gender', 'gender_age'], help="分类任务")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件不存在: {args.audio_file}")
        return
    
    # 创建检测器
    detector = TFAgeGenderDetector(args.model_path, args.cls_task)
    
    # 获取预测结果
    prediction = detector.getPrediction(args.audio_file)
    
    # 打印结果
    print(f"\n音频文件: {args.audio_file}")
    print(f"分类任务: {args.cls_task}")
    
    if args.cls_task == 'gender':
        print(f"预测性别: {prediction}")
    elif args.cls_task == 'age':
        print(f"预测年龄: {prediction}")
    else:  # gender_age
        gender, age = prediction.split('-')
        print(f"预测性别: {gender}")
        print(f"预测年龄: {age}")

if __name__ == '__main__':
    main()
