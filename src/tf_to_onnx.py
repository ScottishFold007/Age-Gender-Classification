'''
# 首先转换模型
!python tf_to_onnx.py --mode convert --tf_model_path gender_age_classification_acc_80.h5 --cls_task gender_age

# 然后使用ONNX模型进行预测
!python tf_to_onnx.py --mode predict --onnx_model_path models/age/age_classification_acc_81.onnx --cls_task age --audio_file path/to/audio.mp3
'''

import os
import tf2onnx
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import librosa
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class ONNXAgeGenderDetector:
    def __init__(self, onnx_model_path, cls_task):
        """
        初始化ONNX模型检测器
        
        参数:
            onnx_model_path: ONNX模型路径
            cls_task: 分类任务类型 ('age', 'gender', 或 'gender_age')
        """
        self.cls_task = cls_task
        
        # 加载ONNX模型
        print(f"加载ONNX模型: {onnx_model_path}")
        self.ort_session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
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
        
        # 加载图像并转换为numpy数组
        from PIL import Image
        img = Image.open(out_file).resize(self._spec_img_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 删除临时文件
        if os.path.exists(out_file):
            os.remove(out_file)
            
        return img_array
    
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
        img_array = self._extractSpectrogram(audio_file)
        
        # 运行推理
        outputs = self.ort_session.run(None, {self.input_name: img_array})
        predicted_label = np.argmax(outputs[0], axis=1)[0]
        
        return self.getPredictionLabelName(predicted_label)


def convert_tf_to_onnx(tf_model_path, onnx_model_path):
    """将TensorFlow模型转换为ONNX格式"""
    print(f"加载TensorFlow模型: {tf_model_path}")
    tf_model = tf.keras.models.load_model(tf_model_path)
    
    print(f"转换为ONNX格式...")
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model, opset=12)
    
    print(f"保存ONNX模型到: {onnx_model_path}")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    return onnx_model_path


def parse_args():
    parser = ArgumentParser(description="TensorFlow模型转换为ONNX并进行推理")
    parser.add_argument("--mode", choices=["convert", "predict"], default="convert",
                        help="模式: convert (转换模型) 或 predict (使用ONNX模型预测)")
    parser.add_argument("--tf_model_path", help="TensorFlow模型路径(.h5文件)")
    parser.add_argument("--onnx_model_path", help="ONNX模型路径")
    parser.add_argument("--cls_task", choices=["age", "gender", "gender_age"],
                        help="分类任务类型")
    parser.add_argument("--audio_file", help="用于预测的音频文件路径(仅在predict模式下需要)")
    
    args = parser.parse_args()
    
    # 设置默认ONNX模型路径
    if args.mode == "convert" and args.tf_model_path and not args.onnx_model_path:
        model_name = os.path.splitext(os.path.basename(args.tf_model_path))[0]
        args.onnx_model_path = f"models/{args.cls_task}/{model_name}.onnx"
        os.makedirs(f"models/{args.cls_task}", exist_ok=True)
    
    return args


def main():
    args = parse_args()
    
    if args.mode == "convert":
        if not args.tf_model_path:
            print("错误: 转换模式需要提供--tf_model_path参数")
            return
        if not args.cls_task:
            print("错误: 转换模式需要提供--cls_task参数")
            return
        
        convert_tf_to_onnx(args.tf_model_path, args.onnx_model_path)
        
    elif args.mode == "predict":
        if not args.onnx_model_path:
            print("错误: 预测模式需要提供--onnx_model_path参数")
            return
        if not args.audio_file:
            print("错误: 预测模式需要提供--audio_file参数")
            return
        if not args.cls_task:
            print("错误: 预测模式需要提供--cls_task参数")
            return
        
        # 检查文件是否存在
        if not os.path.exists(args.onnx_model_path):
            print(f"错误: ONNX模型文件不存在: {args.onnx_model_path}")
            return
        
        if not os.path.exists(args.audio_file):
            print(f"错误: 音频文件不存在: {args.audio_file}")
            return
        
        # 创建检测器
        detector = ONNXAgeGenderDetector(args.onnx_model_path, args.cls_task)
        
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


if __name__ == "__main__":
    main()
