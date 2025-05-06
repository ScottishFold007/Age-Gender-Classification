import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from model_builder import VoiceClassifier

class TensorFlowToPyTorchConverter:
    def __init__(self, tf_model_path, cls_task):
        """
        初始化转换器
        
        参数:
            tf_model_path: TensorFlow模型路径(.h5文件)
            cls_task: 分类任务类型 ('age', 'gender', 或 'gender_age')
        """
        self.tf_model_path = tf_model_path
        self.cls_task = cls_task
        
        # 设置输出PyTorch模型路径
        model_filename = os.path.basename(tf_model_path)
        model_name = os.path.splitext(model_filename)[0]
        self.pytorch_model_path = f"models/{cls_task}/converted_{model_name}.pth"
        
        # 确保输出目录存在
        os.makedirs(f"models/{cls_task}", exist_ok=True)
        
        # 获取类别数量
        if cls_task == 'gender':
            self.num_classes = 2
        elif cls_task == 'age':
            self.num_classes = 6
        else:  # gender_age
            self.num_classes = 12
    
    def load_tensorflow_model(self):
        """加载TensorFlow模型"""
        print(f"加载TensorFlow模型: {self.tf_model_path}")
        self.tf_model = keras.models.load_model(self.tf_model_path)
        self.tf_model.summary()
        return self.tf_model
    
    def create_pytorch_model(self):
        """创建对应的PyTorch模型"""
        print(f"创建PyTorch模型，类别数: {self.num_classes}")
        self.pytorch_model = VoiceClassifier(
            input_shape=(3, 64, 64),
            num_classes=self.num_classes
        )
        return self.pytorch_model
    
    def convert_weights(self):
        """转换模型权重"""
        print("开始转换模型权重...")
        
        # 这里需要根据具体的模型架构进行权重映射
        # 以下是一个简化的示例，实际转换需要更详细的层级映射
        
        # 获取TensorFlow模型的权重
        tf_weights = self.tf_model.get_weights()
        
        # 打印TensorFlow模型的层
        for i, layer in enumerate(self.tf_model.layers):
            try:
                output_shape = layer.output_shape
                print(f"TF层 {i}: {layer.name}, 输出形状: {output_shape}")
            except AttributeError:
                print(f"TF层 {i}: {layer.name}, 输出形状: 无法获取")
        
        # 打印PyTorch模型的参数
        for name, param in self.pytorch_model.named_parameters():
            print(f"PyTorch参数: {name}, 形状: {param.shape}")
        
        # 这里需要手动映射TensorFlow和PyTorch模型的层
        # 由于模型架构复杂，这需要详细的层级对应关系
        
        print("警告: 完整的权重转换需要详细的层级映射，这超出了简单示例的范围")
        print("建议: 使用ONNX作为中间格式进行转换，或者重新训练PyTorch模型")
        
        return self.pytorch_model
    
    def save_pytorch_model(self):
        """保存转换后的PyTorch模型"""
        print(f"保存PyTorch模型到: {self.pytorch_model_path}")
        torch.save(self.pytorch_model.state_dict(), self.pytorch_model_path)
        return self.pytorch_model_path


class TensorFlowModelWrapper:
    """包装TensorFlow模型，使其可以在PyTorch代码中使用"""
    
    def __init__(self, tf_model_path, cls_task):
        """
        初始化TensorFlow模型包装器
        
        参数:
            tf_model_path: TensorFlow模型路径(.h5文件)
            cls_task: 分类任务类型 ('age', 'gender', 或 'gender_age')
        """
        self.tf_model_path = tf_model_path
        self.cls_task = cls_task
        
        # 加载TensorFlow模型
        self.model = keras.models.load_model(tf_model_path)
        
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
        
        print(f"已加载TensorFlow模型: {tf_model_path}")
    
    def scale_minmax(self, X, min=0, max=255):
        """缩放数组到指定范围"""
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
    
    def saveSpectrogram(self, data, fn):
        """保存频谱图为图像文件"""
        import matplotlib.pyplot as plt
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
        import librosa
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


# 使用示例
def convert_model(tf_model_path, cls_task):
    """转换TensorFlow模型到PyTorch"""
    converter = TensorFlowToPyTorchConverter(tf_model_path, cls_task)
    converter.load_tensorflow_model()
    converter.create_pytorch_model()
    converter.convert_weights()
    pytorch_model_path = converter.save_pytorch_model()
    print(f"转换完成，PyTorch模型保存在: {pytorch_model_path}")
    return pytorch_model_path

def use_tf_model_directly(tf_model_path, cls_task, audio_file):
    """直接使用TensorFlow模型进行预测"""
    wrapper = TensorFlowModelWrapper(tf_model_path, cls_task)
    prediction = wrapper.getPrediction(audio_file)
    print(f"音频文件: {audio_file}")
    print(f"预测结果: {prediction}")
    return prediction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorFlow模型转换和使用工具")
    parser.add_argument("--mode", choices=["convert", "predict"], default="predict",
                        help="模式: convert (转换模型) 或 predict (直接使用TF模型预测)")
    parser.add_argument("--model_path", required=True, help="TensorFlow模型路径(.h5文件)")
    parser.add_argument("--cls_task", required=True, choices=["age", "gender", "gender_age"],
                        help="分类任务类型")
    parser.add_argument("--audio_file", help="用于预测的音频文件路径(仅在predict模式下需要)")
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        convert_model(args.model_path, args.cls_task)
    elif args.mode == "predict":
        if not args.audio_file:
            parser.error("predict模式需要提供--audio_file参数")
        use_tf_model_directly(args.model_path, args.cls_task, args.audio_file)
