import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import logging
from datetime import datetime
from tqdm import tqdm
from pytorch_model_builder import VoiceClassifier
import librosa
from PIL import Image
import torchvision.transforms as transforms

class ModelTrainer:
    def __init__(self, dataset_dir, cls_task):
        self.dataset_dir = dataset_dir
        self.cls_task = cls_task
        self.reg_value = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_symbols = 150
        self._print_symbol = '='
        
        # 创建保存目录
        os.makedirs(f'results/{self.cls_task}', exist_ok=True)
        os.makedirs(f'models/{self.cls_task}', exist_ok=True)
        
        self.log_file = f'results/{self.cls_task}/train_logs_{self.cls_task}.csv'
        self.best_model_filepath = f'models/{self.cls_task}/best_model_{self.cls_task}.pth'
        
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
    
    def getBestModelFilepath(self):
        return self.best_model_filepath
    
    def setTrainingParameters(self, batch_size, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def _buildDataset(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 获取类别数量
        if self.cls_task == 'gender':
            self.num_classes = 2
        elif self.cls_task == 'age':
            self.num_classes = 6
        else:  # gender_age
            self.num_classes = 12
    
    def _buildModel(self, show_summary=False):
        # 假设输入图像大小为64x64，3通道
        self.model = VoiceClassifier(input_shape=(3, 64, 64), 
                                    num_classes=self.num_classes, 
                                    reg_value=self.reg_value).to(self.device)
        
        if show_summary:
            self.model.summary()
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                                   weight_decay=self.reg_value)
    
    def startTraining(self, train_loader, val_loader, show_model_summary=False, show_live_plot=False):
        self._buildDataset(train_loader, val_loader)
        self._buildModel(show_summary=show_model_summary)
        
        # 初始化训练监控
        if show_live_plot:
            self.model_monitoring = ModelTrainingMonitoring(self.log_file, self.cls_task)
        
        # 初始化日志
        log_data = {
            'epoch': [], 
            'loss': [], 
            'accuracy': [], 
            'val_loss': [], 
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            self.printTitle(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(self.train_loader, desc="Training") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # 梯度清零
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 统计
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': train_loss / (pbar.n + 1),
                        'acc': 100. * train_correct / train_total
                    })
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                with tqdm(self.val_loader, desc="Validation") as pbar:
                    for inputs, labels in pbar:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        
                        # 统计
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'loss': val_loss / (pbar.n + 1),
                            'acc': 100. * val_correct / val_total
                        })
            
            val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 更新日志
            log_data['epoch'].append(epoch + 1)
            log_data['loss'].append(train_loss)
            log_data['accuracy'].append(train_acc)
            log_data['val_loss'].append(val_loss)
            log_data['val_accuracy'].append(val_acc)
            
            # 保存日志
            pd.DataFrame(log_data).to_csv(self.log_file, index=False)
            
            # 更新监控图
            if show_live_plot:
                self.model_monitoring.update(log_data)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_filepath)
                print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 训练结束，保存最终图表
        if show_live_plot:
            self.model_monitoring.save_figure()
        
        print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")


class ModelEvaluator:
    def __init__(self, dataset_dir, cls_task, model_path):
        self.dataset_dir = dataset_dir
        self.cls_task = cls_task
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化标签解码器
        self._decode_gender_age = {0:'female-teens', 1:'female-twenties', 2:'female-thirties', 3:'female-fourties', 4:'female-fifties', 5:'female-sixties',  
                                6:'male-teens', 7:'male-twenties', 8:'male-thirties', 9:'male-fourties', 10:'male-fifties', 11:'male-sixties'}
        self._decode_gender = {0:'male', 1:'female'}
        self._decode_age = {0:'teens', 1:'twenties', 2:'thirties', 3:'fourties', 4:'fifties', 5:'sixties'}
        
        # 设置输出文件路径
        self.cm_plot_file = f'results/{self.cls_task}/confusion_matrix_{self.cls_task}.png'
        self.test_log_file = f'results/{self.cls_task}/test_log_{self.cls_task}.txt'
        
        self._max_symbols = 150
        self._print_symbol = '='
    
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        message = num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol
        print(message)
        return message
    
    def get_category_names(self):
        if self.cls_task == 'gender':
            return list(self._decode_gender.values())
        elif self.cls_task == 'age':
            return list(self._decode_age.values())
        else:  # gender_age
            return list(self._decode_gender_age.values())
    
    def evaluateBestModel(self, test_loader):
        message = self.printTitle(f" 在测试数据集上评估最佳模型 ({self.cls_task} 分类) ")
        
        # 获取类别数量
        if self.cls_task == 'gender':
            num_classes = 2
        elif self.cls_task == 'age':
            num_classes = 6
        else:  # gender_age
            num_classes = 12
        
        # 加载模型
        model = VoiceClassifier(input_shape=(3, 64, 64), 
                               num_classes=num_classes).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        
        # 获取类别名称
        self.category_names = self.get_category_names()
        
        # 评估模型
        labels = []
        predictions = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="测试中"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                labels.extend(targets.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
        
        # 计算评估指标
        acc = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        clr = classification_report(labels, predictions, target_names=self.category_names)
        
        # 保存评估结果
        with open(self.test_log_file, 'w') as f:
            message = self.printTitle(" 准确率 ")
            f.write(message + '\n')
            
            message = "{:.2f}".format(acc)
            f.write(message + '\n')
            print(message)
            
            message = self.printTitle(" 混淆矩阵 ")
            f.write(message + '\n')
            
            print(cm)
            for row in cm:
                row_str = '\t'.join(map(str, row))
                f.write(row_str + '\n')
            
            message = self.printTitle(" 分类报告 ")
            f.write(message + '\n')
            
            print(clr)
            f.write(clr)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, target_names=self.category_names, filename=self.cm_plot_file)
    
    def plot_confusion_matrix(self, cm, target_names, filename, title='混淆矩阵', normalize=True):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm *= 100
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', 
                         xticklabels=target_names, yticklabels=target_names, 
                         linecolor='white', linewidths=.5)
        ax.set(xlabel='预测标签', ylabel='真实标签')
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class AgeGenderDetector:
    def __init__(self, model_path, cls_task):
        self.cls_task = cls_task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取类别数量
        if cls_task == 'gender':
            num_classes = 2
        elif cls_task == 'age':
            num_classes = 6
        else:  # gender_age
            num_classes = 12
        
        # 加载模型
        self.model = VoiceClassifier(input_shape=(3, 64, 64), 
                                    num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
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
        
        # 设置图像转换
        self.transform = transforms.Compose([
            transforms.Resize(self._spec_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def scale_minmax(self, X, min=0, max=255):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
    
    def saveSpectrogram(self, data, fn):
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        fig = plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _extractSpectrogram(self, audio_file):
        y, sr = librosa.load(audio_file, sr=self.sampling_rate)
        spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
        spec = librosa.amplitude_to_db(np.abs(spec))
        # min-max scale to fit inside 8-bit range
        img = self.scale_minmax(spec).astype(np.uint8)
        out_file = 'temp.png'
        self.saveSpectrogram(img, out_file)
        return out_file
    
    def getPredictionLabelName(self, encod_lab):
        if self.cls_task == 'age':
            return self._decode_age.get(encod_lab)
        elif self.cls_task == 'gender':
            return self._decode_gender.get(encod_lab)
        elif self.cls_task == 'gender_age':
            return self._decode_gender_age.get(encod_lab)
    
    def getPrediction(self, audio_file):
        # 提取频谱图
        spec_file = self._extractSpectrogram(audio_file)
        
        # 加载图像并应用转换
        image = Image.open(spec_file).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()
        
        # 删除临时文件
        if os.path.exists(spec_file):
            os.remove(spec_file)
        
        return self.getPredictionLabelName(predicted_label)


class ModelTrainingMonitoring:
    def __init__(self, log_file, cls_task):
        plt.rcParams['figure.figsize'] = (18, 4)
        plt.ion()
        self.cls_task = cls_task
        self._loss_title = f'Loss monitoring for {self.cls_task} classification'
        self._acc_title = f'Accuracy monitoring for {self.cls_task} classification'
        self._fig_title = f"Training metrics monitoring for {cls_task} classification"
        self.main_fig = plt.figure(self._fig_title, constrained_layout=True)
        self.gridspec = GridSpec(1, 2, figure=self.main_fig)
        self.ax_acc = self.get_subplot(self.gridspec[0, 0], title=self._acc_title)
        self.ax_loss = self.get_subplot(self.gridspec[0, 1], title=self._loss_title)
    
    def get_subplot(self, grid, title):
        ax = self.main_fig.add_subplot(grid)
        ax.set_title(title)
        return ax
    
    def update(self, log_data):
        epochs = log_data['epoch']
        tr_loss = log_data['loss']
        tr_acc = log_data['accuracy']
        val_loss = log_data['val_loss']
        val_acc = log_data['val_accuracy']
        
        self.plotAccuracy(epochs, tr_acc, val_acc)
        self.plotLoss(epochs, tr_loss, val_loss)
        plt.draw()
        plt.pause(0.001)
    
    def plotLoss(self, epochs, tr_loss, val_loss):
        self.ax_loss.cla()
        self.ax_loss.set_title(self._loss_title)
        self.ax_loss.plot(epochs, tr_loss, label='Train loss')
        self.ax_loss.scatter(epochs[-1], tr_loss[-1], s=10)
        self.ax_loss.plot(epochs, val_loss, label='Validation loss')
        self.ax_loss.scatter(epochs[-1], val_loss[-1], s=10)
        self.ax_loss.set_xlabel('Number of epochs')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.set_aspect('auto')
        self.ax_loss.grid()
    
    def plotAccuracy(self, epochs, tr_acc, val_acc):
        self.ax_acc.cla()
        self.ax_acc.set_title(self._acc_title)
        self.ax_acc.plot(epochs, tr_acc, label='Train accuracy')
        self.ax_acc.scatter(epochs[-1], tr_acc[-1], s=10)
        self.ax_acc.plot(epochs, val_acc, label='Validation accuracy')
        self.ax_acc.scatter(epochs[-1], val_acc[-1], s=10)
        self.ax_acc.set_xlabel('Number of epochs')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.legend(loc='lower right')
        self.ax_acc.set_aspect('auto')
        self.ax_acc.grid()
    
    def save_figure(self):
        plt.savefig(f"results/{self.cls_task}/training_metrics_{self.cls_task}.png", bbox_inches='tight')
        plt.close()
