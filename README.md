## Age-Gender-Classification


本文《通过语音频谱图与特殊设计的多重注意力模块实现基于卷积神经网络的年龄与性别识别》的官方实现代码。本实现针对Common Voice数据集开发，但可适配至任何自定义数据集。论文可从[此处](https://www.mdpi.com/1424-8220/21/17/5892)下载。

引言  
----------------------------  
语音信号作为人机交互(HCI)的主要输入源，已被用于开发自动语音识别(ASR)、语音情感识别(SER)、性别与年龄识别等多种应用。由于现有方法在提取显著高层语音特征和分类模型方面存在局限，根据年龄和性别对说话人进行分类成为语音处理领域的一项挑战性任务。为解决这些问题，我们提出了一种新颖的端到端年龄性别识别卷积神经网络(CNN)，其配备特殊设计的多重注意力模块(MAM)，可直接处理语音信号。  

我们提出的模型利用MAM从输入数据中高效提取空间与时间显著特征。该MAM机制采用矩形滤波器作为卷积层内核，包含独立的时间注意力和频率注意力分支。时间注意力分支学习检测时序线索，而频率注意力模块则通过聚焦空间频率特征来提取与目标最相关的特征。两种提取的空间与时间特征相互补充，为年龄和性别分类提供了高性能解决方案。  

该年龄性别分类系统在Common Voice数据集和本地开发的韩语语音识别数据集上进行了测试。在Common Voice数据集上，我们的模型在性别、年龄及年龄-性别联合分类任务中分别达到96%、73%和76%的准确率。在韩语语音识别数据集上，三项任务的准确率分别为97%、97%和90%。实验结果表明，我们提出的模型在语音信号的年龄、性别及联合识别任务中具有卓越的鲁棒性和优越性。  


<img width="1293" alt="image" src='figs/proposed_framework.png'>

### 音频性别和年龄识别系统流程图


[![](https://mermaid.ink/img/pako:eNqNlVtP20gUx79KNE-LFCCOc4E8rEScyxPVSq360HUfvMkQIiV25Di720VIvYRLoU1gFwpqCmRRWbJbEagoUC4JXyYzdr5FZ3zcJFbaFX6aI_995v87Z-Z4DqW0NEYRNJPTfkvNKrrheRCTVQ97iqVfMrpSmPXIiG6e0NfN7n6ZvC-ba4vd7XPaPJMR6PgzJfwsI0nL5zX1oZZN4ZhiKEVs_KTjgq6lcLGo6TJ67Bkd_dEz5WdSslK3Wi1S2SOHq07y2iJTDGT0g1pkavp5ge1Kt-p0cxnUbqkI0gCTdm53yNE22CRrFfNtubt32t1fJ3--ovUl86jd-xKraVgM0ZovL0n7Ga2ukeqbYdQoR71fwClD19gX-SRWsa4YfcAoB-xcV2Bj-mapc30OySCxLKs_sBfWyUdSa4_T_QVystGPpxOSNOLCi0IlomIPr_P5iknJiyqt35CbKk9objQ6VxXp3j3a-Jvsrlrtv8jCwcgdYHvVJ8s1cn01zCtx3sGmfuWU-o0EMPDk8i6Bd4l7J1cbzCXsR6rb9PW-WwpdlHgXwYvVPDKvX4x3_3tlHT8bp2er1vHm4DH5Hygogv39MFGsRyTllGIxO5PFvebFOJSDc_RP57JOdrYEl88YIMVspPdv6WmDlqtkpcb33NlySwEpFhjO6XcLAyAM8pwLDet2l1YOyMfnLlF_FecA0-ze5h7oSlbtu49z91azbbaaAE-qz1nBXWni4D8u2lIugmq5ReA83r9S9N3TTuv0W1LwHg-6PGXVzLSmZtm1YCt-QElzl26dm7V1WjmEba3bJbOxOnL3dh6XOzcnw-1M9KoR_1XJlQavYqJ_RL_rPwH1SNj9fNfoHTPYzi2FqiTsfm7s0eU1enFBLxbNvX-722d8xKwckPWVuxBVGt-co0nOMpXBbKiksR7DBhszfZxkH2dwuLg8JgEnac_NwanjEgFI0m5v65ZfS7swbMIzfvr0kCx_GCeXn7rt8vdgpqDtUQHCqHN7nVByDrQ7jDuhc2QSTpiAMCkgL8pjPa9k0-yHNMdfysiYxXksowhbpvGMUsoZvF7zTKqUDO3-EzWFIoZewl6ka6XMLIrMKLkii0qFtGLgWFbhM_qrpKCojzStF-I0P6DT8AO0_4O2BEXm0O8oIoSCYyFfaFIMhX2TYX9wwoueoMioIApjvgnRFwz4wuGJoBAIz3vRH3ZWYcwfCk34_KLgmwyIIZ8Y9KKMzmkch3ZXJa2kGigyOf8FkpcDsg?type=png)](https://mermaid-live.nodejs.cn/edit#pako:eNqNlVtP20gUx79KNE-LFCCOc4E8rEScyxPVSq360HUfvMkQIiV25Di720VIvYRLoU1gFwpqCmRRWbJbEagoUC4JXyYzdr5FZ3zcJFbaFX6aI_995v87Z-Z4DqW0NEYRNJPTfkvNKrrheRCTVQ97iqVfMrpSmPXIiG6e0NfN7n6ZvC-ba4vd7XPaPJMR6PgzJfwsI0nL5zX1oZZN4ZhiKEVs_KTjgq6lcLGo6TJ67Bkd_dEz5WdSslK3Wi1S2SOHq07y2iJTDGT0g1pkavp5ge1Kt-p0cxnUbqkI0gCTdm53yNE22CRrFfNtubt32t1fJ3--ovUl86jd-xKraVgM0ZovL0n7Ga2ukeqbYdQoR71fwClD19gX-SRWsa4YfcAoB-xcV2Bj-mapc30OySCxLKs_sBfWyUdSa4_T_QVystGPpxOSNOLCi0IlomIPr_P5iknJiyqt35CbKk9objQ6VxXp3j3a-Jvsrlrtv8jCwcgdYHvVJ8s1cn01zCtx3sGmfuWU-o0EMPDk8i6Bd4l7J1cbzCXsR6rb9PW-WwpdlHgXwYvVPDKvX4x3_3tlHT8bp2er1vHm4DH5Hygogv39MFGsRyTllGIxO5PFvebFOJSDc_RP57JOdrYEl88YIMVspPdv6WmDlqtkpcb33NlySwEpFhjO6XcLAyAM8pwLDet2l1YOyMfnLlF_FecA0-ze5h7oSlbtu49z91azbbaaAE-qz1nBXWni4D8u2lIugmq5ReA83r9S9N3TTuv0W1LwHg-6PGXVzLSmZtm1YCt-QElzl26dm7V1WjmEba3bJbOxOnL3dh6XOzcnw-1M9KoR_1XJlQavYqJ_RL_rPwH1SNj9fNfoHTPYzi2FqiTsfm7s0eU1enFBLxbNvX-722d8xKwckPWVuxBVGt-co0nOMpXBbKiksR7DBhszfZxkH2dwuLg8JgEnac_NwanjEgFI0m5v65ZfS7swbMIzfvr0kCx_GCeXn7rt8vdgpqDtUQHCqHN7nVByDrQ7jDuhc2QSTpiAMCkgL8pjPa9k0-yHNMdfysiYxXksowhbpvGMUsoZvF7zTKqUDO3-EzWFIoZewl6ka6XMLIrMKLkii0qFtGLgWFbhM_qrpKCojzStF-I0P6DT8AO0_4O2BEXm0O8oIoSCYyFfaFIMhX2TYX9wwoueoMioIApjvgnRFwz4wuGJoBAIz3vRH3ZWYcwfCk34_KLgmwyIIZ8Y9KKMzmkch3ZXJa2kGigyOf8FkpcDsg)


#### 1. 数据预处理阶段
- **CommonVoiceDatasetPreprocessor**：处理原始的Common Voice数据集
  - 加载CSV文件中的音频元数据
  - 清理无效数据（如缺失标签、非目标性别/年龄组）
  - 将处理后的音频文件和标签保存到指定目录

#### 2. 特征提取阶段
- **SpectrogramGenerator**：从音频文件生成特征图像
  - 支持多种特征类型：频谱图、梅尔频谱图、MFCC
  - 将特征转换为图像格式并保存
  - 特征图像将作为CNN模型的输入

#### 3. 数据集创建阶段
- **VoiceDataset**：创建PyTorch数据集
  - 加载特征图像和对应标签
  - 应用数据变换（调整大小、归一化等）
  - 创建训练集、验证集和测试集

#### 4. 模型训练阶段
- **VoiceClassifier**：定义模型架构
  - 特征学习块：提取音频特征
  - 多注意力模块：关注时间和频率维度的特征
  - 全连接层：进行最终分类
- **ModelTrainer**：训练模型
  - 设置训练参数（批次大小、学习率等）
  - 训练模型并监控性能
  - 保存最佳模型

#### 5. 模型评估阶段
- **ModelEvaluator**：评估模型性能
  - 在测试集上评估模型
  - 生成混淆矩阵和分类报告
  - 保存评估结果

#### 6. 推理阶段
- **AgeGenderDetector**：使用训练好的模型进行推理
  - 加载新的音频文件
  - 提取特征
  - 使用模型预测性别和/或年龄

这个流程图展示了整个系统从原始音频数据到最终模型预测的完整过程，包括数据处理、特征提取、模型训练和评估的各个步骤。


Installation
------------------------------
所提出的方法基于 Torch 在 Ubuntu 操作系统上实现，建议安装 GPU 版本的 Torch 以获得最佳性能。

### Example conda environment setup
```bash
conda create --name ag_cls python=3.8 -y
conda activate ag_cls
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
pip install -r requirements.txt
```

Usage instructions
----------------------------------
首先需要下载 Common Voice 数据集并放入 `data/` 文件夹。您可以选择下载[原始数据集](https://www.kaggle.com/datasets/mozillaorg/common-voice/data)或[清洗版数据集](https://1drv.ms/u/s!AtLl-Rpr0uJohKI71cVKzwcbo76FVg?e=vAar5G)，建议下载**清洗版**数据集。下载完成后将清洗版数据集解压至 `data/` 文件夹，目录结构应如下所示：

```bash
├──data/
│  └──CommonVoice/
│     ├──Audio/
│     │  ├──cv-valid-dev/
│     │  ├──cv-valid-test/
│     │  ├──cv-valid-train/
│     │  ├──LICENSE.txt
│     │  ├──README.txt
│     │  ├──cv-valid-dev.csv
│     │  ├──cv-valid-test.csv
│     │  └──cv-valid-train.csv
```

Dataset preparation
-------------------------------
数据集准备包含两个主要步骤：  
1. 为每个音频文件生成频谱图，并保存至 `data/CommonVoice/Spectrograms/` 文件夹  
2. 创建 `tfrecord` 格式数据集并保存至 `data/CommonVoice/TFRecord/` 文件夹  

执行以下操作准备数据集：  
使用 `src/preprocess_dataset.py` Python 脚本，在项目主目录的 PowerShell 终端中运行如下命令：  

```bash
# 若预处理的是原始数据集（raw），需指定 --dataset_type='raw'
python .\src\preprocess_dataset.py --dataset_dir='data/CommonVoice/' --dataset_type='clean'
```  

完成数据集预处理后，`data/` 文件夹结构将如下所示：  


```bash
├───data/
│   └───CommonVoice/
│       ├───Audio/
│       │   ├───cv-valid-dev/
│       │   │   ├───sample-000004.mp3
│       │   │   ├───sample-000005.mp3
│       │   ├───cv-valid-test/
│       │   │   ├───sample-000001.mp3
│       │   │   ├───sample-000003.mp3
│       │   ├───cv-valid-train/
│       │   │   ├───sample-000005.mp3
│       │   │   ├───sample-000013.mp3
│       │   ├───LICENSE.txt
│       │   ├───README.txt
│       │   ├───cv-valid-dev.csv
│       │   ├───cv-valid-test.csv
│       │   └───cv-valid-train.csv
│       ├───Spectrograms/
│       │   ├───cv-valid-dev/
│       │   │   ├───sample-000004.png
│       │   │   ├───sample-000005.png
│       │   ├───cv-valid-test/
│       │   │   ├───sample-000001.png
│       │   │   ├───sample-000003.png
│       │   ├───cv-valid-train/
│       │   │   ├───sample-000005.png
│       │   │   ├───sample-000013.png
│       ├───TFRecord/
│       │   ├───cv-valid-dev/
│       │   │   ├───spec_data_000.tfrecord
│       │   │   ├───spec_data_001.tfrecord
│       │   ├───cv-valid-test/
│       │   │   ├───spec_data_000.tfrecord
│       │   │   ├───spec_data_001.tfrecord
│       │   └───cv-valid-train/
│       │   │   ├───spec_data_000.tfrecord
│       │   │   ├───spec_data_001.tfrecord
```

Model training
-------------------------------
要训练年龄、性别或年龄-性别分类模型，可使用 `src/train.py` Python 脚本。以下为使用 `train.py` 脚本的示例代码：

```bash
python .\src\train.py --dataset_dir='data/CommonVoice' --cls_task='age' --num_epochs=20 --show_live_plot=True
```

您可指定其他训练参数，例如 `--batch_size` 和 `--learning_rate`。`--batch_size` 默认值为 128，`--learning_rate` 默认值为 0.0001。另有 `--show_summary` 参数用于显示模型摘要。`--show_summary` 和 `--show_live_plot` 的默认值均为 False。训练完成后，训练日志文件将根据分类任务保存至 `results/` 文件夹，最佳模型将保存至 `models/` 文件夹。
Pretrained models 
--------------------------------
pretrained models can be downloaded from [here](https://1drv.ms/u/s!AtLl-Rpr0uJohKJ6_236uKDuJsLkhA?e=7zmPvM)

**Note:** 预训练模型的性能优于论文中报告的结果，这是因为这些模型是近期使用稍加调整的训练参数重新训练的。

Model testing
-------------------------------
要测试训练好的模型，可以使用 `src/test.py` Python 脚本。使用方法如下：

```bash
python .\src\test.py --dataset_dir='data/CommonVoice' --model_path='models/age/best_model_age.h5' --cls_task='age'
```

Inference code
-------------------------------
要使用训练好的模型测试单个音频文件，可以使用 `src/inference.py` Python 脚本。示例如下：

```bash
python .\src\inference.py --model_path='models/age/best_model_age.h5' --audio_file='data/CommonVoice/Audio/cv-valid-test/sample-000001.mp3' --cls_task='age'
```
