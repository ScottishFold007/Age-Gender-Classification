import os
import os.path as osp
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from shutil import move, rmtree
from tabulate import tabulate
from glob import glob
from PIL import Image
from alive_progress import alive_bar
import torch
from torch.utils.data import Dataset, DataLoader

class CommonVoiceDatasetPreprocessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self._invalid_gender = 'other'
        self._major_gender_age_groups = ['male-teens', 'male-twenties', 'male-thirties', 'male-fourties', 'male-fifties', 'male-sixties', 
                                         'female-teens', 'female-twenties', 'female-thirties', 'female-fourties', 'female-fifties', 'female-sixties']
        self._minor_gender_age_groups = ['male-seventies', 'male-eighties', 'female-seventies', 'female-eighties']
        self._modes = ['train', 'dev', 'test']
        self._csv_filenames = ['cv-valid-train.csv', 'cv-valid-dev.csv', 'cv-valid-test.csv']
        self._keep_keyword = '-valid-'
        self._max_symbols = 150
        self._print_symbol = '='
    
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
    
    def printPrettyTable(self, dataframe, only_head=True):
        if only_head:
            print(tabulate(dataframe.head(n=3), headers='keys', tablefmt='fancy_grid', showindex=False, stralign='left', numalign='center', maxcolwidths=35))
        else:
            print(tabulate(dataframe, headers='keys', tablefmt='fancy_grid', showindex=False, stralign='left', numalign='center', maxcolwidths=35))
    
    def removeUnnecessaryFolderAndFiles(self):
        self.printTitle('Removing unnecessary folder and files')
        all_folders = [fol for fol in os.listdir(self.dataset_dir) if osp.isdir(osp.join(self.dataset_dir, fol))]
        all_csv_files = glob(self.dataset_dir + '/*.csv')
        print('Folder names: ')
        for fol in all_folders:
            if self._keep_keyword not in fol:
                print(osp.basename(fol))
                path = osp.join(osp.join(self.dataset_dir, fol), fol)
                self.deleteFiles(path)
        print('\nFile names: ')
        for fn in all_csv_files:
            if self._keep_keyword not in fn:
                print(osp.basename(fn))
                os.remove(fn)
    
    def deleteFiles(self, path):
        if os.path.exists(path):
            filenames = os.listdir(path)
            n = len(filenames)
            with alive_bar(n) as bar:
                for fn in filenames:
                    os.remove(osp.join(path, fn))
                    bar()
            os.removedirs(path)
        else:
            print(f"路径不存在: {path}")
    
    def loadDataset(self, mode):
        if mode in self._modes:
            if mode == 'train':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[0]))
                return dataframe
            elif mode == 'dev':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[1]))
                return dataframe
            elif mode == 'test':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[2]))
                return dataframe
        else:
            print(f'Invalid mode is given. Mode should be one of {self._modes} modes')
    
    def cleanDataset(self, dataframe):
        self.printTitle('Raw dataset information')
        self.printPrettyTable(dataframe)
        print(f'Number of files ===> {len(dataframe)}')
        ag_df = dataframe[['filename', 'age', 'gender']].copy()
        self.printTitle('Dataset information after removing unnecessary columns')
        self.printPrettyTable(ag_df)
        print(f'Number of files ===> {len(ag_df)}')
        ag_df.dropna(axis=0, inplace=True)
        print(f'Number of files after removing rows with NaN values ===> {len(ag_df)}')
        print(f"Age groups ===> {list(ag_df['age'].unique())}")
        print(f"Gender groups ===> {list(ag_df['gender'].unique())}")
        ag_df = ag_df[ag_df['gender'] != self._invalid_gender]
        print(f'Number of files after removing rows with `{self._invalid_gender}` gender group values ===> {len(ag_df)}')
        ag_df['gender_age'] = ag_df['gender'] + '-' + ag_df['age']
        # Calculate total entries for each gender
        total_males = len(ag_df[ag_df['gender'] == 'male'])
        total_females = len(ag_df[ag_df['gender'] == 'female'])
        # Calculate value counts for each 'gender_age' group
        value_counts = ag_df['gender_age'].value_counts()
        # Calculate percentages based on total entries for each gender
        percentages = {}
        for group, count in value_counts.items():
            gender = group.split('-')[0]
            if gender == 'male':
                percentages[group] = (count / total_males) * 100
            elif gender == 'female':
                percentages[group] = (count / total_females) * 100

        # Create a DataFrame to display the results
        result_df = pd.DataFrame({'Gender-Age group': value_counts.index,'Count': value_counts.values, 'Percentage': percentages.values()})
        result_df = result_df.sort_values(by='Percentage', ascending=False)
        self.printTitle('Dataset statistical information for gender-age groups')
        self.printPrettyTable(result_df, only_head=False)
        # Filter the DataFrame based on valid 'gender_age' groups
        ag_df = ag_df[ag_df['gender_age'].isin(self._major_gender_age_groups)]
        self.printTitle(f'Removing minority {self._minor_gender_age_groups} group files')
        print(f'Final number of files: {len(ag_df)}')
        return ag_df
    
    def writeCleanedDataset(self, mode):
        self.printTitle(f'Started writing clean {mode} dataset')
        self._audio_dir = osp.join(self.dataset_dir, 'Audio')
        folder_name = 'cv-valid-' + mode
        src_dir = osp.join(self.dataset_dir, folder_name)
        des_dir = osp.join(self._audio_dir, folder_name)
        if not osp.exists(des_dir):
            os.makedirs(des_dir)
        
        filenames = self._dataframe['filename'].to_list()
        n = len(filenames)
        with alive_bar(n) as bar:
            for fn in filenames:
                move(osp.join(src_dir, fn), des_dir)
                bar()
        self._dataframe.to_csv(osp.join(self._audio_dir, folder_name + '.csv'), index=False)
        self.printTitle(f'Deleting audio files with invalid labels for {mode} dataset')
        self.deleteFiles(osp.join(src_dir, folder_name))
        os.remove(osp.join(self.dataset_dir, folder_name + '.csv'))
    
    def startPreprocessing(self):
        self.removeUnnecessaryFolderAndFiles()
        for i, mode in enumerate(self._modes[1:]):
            self.printTitle(f'Preprocessing started for {mode} files')
            print(f'Filename ===> {self._csv_filenames[i]}')
            self._dataframe = self.loadDataset(mode)
            self._dataframe = self.cleanDataset(self._dataframe)
            self.writeCleanedDataset(mode)
        move(osp.join(self.dataset_dir, 'LICENSE.txt'), self._audio_dir)
        move(osp.join(self.dataset_dir, 'README.txt'), self._audio_dir)
        

class SpectrogramGenerator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.audio_dir = osp.join(dataset_dir, 'Audio')
        self.spectrogram_dir = osp.join(dataset_dir, 'Spectrograms')
        os.makedirs(self.spectrogram_dir, exist_ok=True)
        self._modes = ['train', 'dev', 'test']
        # setting the default parameters for Spectrogram generation
        self.sampling_rate = 16000
        self.n_fft = 256
        self.num_overlap = 128
        self._max_symbols = 150
        self._print_symbol = '='
        
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
    
    def scale_minmax(self, X, min=0, max=255):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def saveSpectrogram(self, data, fn):
        plt.axis('off')
        fig = plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()

    def startExtractingSpectrograms(self):
        for mode in self._modes:
            self.printTitle(f'Spectrogram generation started for {mode} dataset')
            folder_name = 'cv-valid-' + mode
            self.out_dir = osp.join(self.spectrogram_dir, folder_name)
            os.makedirs(self.out_dir, exist_ok=True)
            audio_files_dir = osp.join(self.audio_dir, folder_name)
            audio_files = glob(os.path.join(audio_files_dir, "*.mp3"))
            n = len(audio_files)
            print(f'Reading audio files from ==> {audio_files_dir}')
            print(f'Number of audio files ==> {n}')
            print(f'Saving extracted spectrograms to ==> {self.out_dir}\n')
            with alive_bar(n) as bar:
                for audio_file in audio_files:
                    f_name = osp.basename(audio_file).replace('.mp3', '.png')
                    y, sr = librosa.load(audio_file, sr=self.sampling_rate)
                    spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
                    spec = librosa.amplitude_to_db(np.abs(spec))
                    # min-max scale to fit inside 8-bit range
                    img = self.scale_minmax(spec).astype(np.uint8)
                    out_file = osp.join(self.out_dir, f_name)
                    self.saveSpectrogram(img, out_file)
                    bar()


class VoiceDataset(Dataset):
    def __init__(self, dataset_dir, mode, cls_task='gender_age', transform=None):
        self.dataset_dir = dataset_dir
        self.audio_dir = osp.join(dataset_dir, 'Audio')
        self.spec_dir = osp.join(dataset_dir, 'Spectrograms')
        self.mode = mode
        self.cls_task = cls_task
        self.transform = transform
        
        # 初始化标签编码器
        self._encode_gender_age = {'female-teens': 0, 'female-twenties': 1, 'female-thirties': 2, 'female-fourties': 3, 'female-fifties': 4, 'female-sixties': 5,  
                                'male-teens': 6, 'male-twenties': 7, 'male-thirties': 8, 'male-fourties': 9, 'male-fifties': 10, 'male-sixties': 11}
        self._encode_gender = {'male': 0, 'female': 1}
        self._encode_age = {'teens': 0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5}
        
        # 加载数据集
        folder_name = f'cv-valid-{mode}'
        csv_path = osp.join(self.audio_dir, f'{folder_name}.csv')
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = osp.join(self.spec_dir, f"cv-valid-{self.mode}", row['filename'].replace('.mp3', '.png'))
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        if self.cls_task == 'gender':
            label = self._encode_gender[row['gender']]
        elif self.cls_task == 'age':
            label = self._encode_age[row['age']]
        else:  # gender_age
            label = self._encode_gender_age[row['gender_age']]
            
        return image, label
    
    def get_num_classes(self):
        if self.cls_task == 'gender':
            return len(self._encode_gender)
        elif self.cls_task == 'age':
            return len(self._encode_age)
        else:  # gender_age
            return len(self._encode_gender_age)
    
    def get_class_names(self):
        if self.cls_task == 'gender':
            return list(self._encode_gender.keys())
        elif self.cls_task == 'age':
            return list(self._encode_age.keys())
        else:  # gender_age
            return list(self._encode_gender_age.keys())


# 使用示例
def get_dataloaders(dataset_dir, batch_size=32, cls_task='gender_age'):
    from torchvision import transforms
    
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
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
