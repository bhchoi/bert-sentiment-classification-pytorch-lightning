import torch
from torch.utils.data import Dataset
import pandas as pd


class TextDataset(Dataset):
    """nsmc dataset"""
    
    def __init__(self, file_path: str, preprocessor: 'Preprocessor', max_len: int):
        """nsmc dataset을 초기화한다.

        Args:
            file_path (str): 파일 경로
            preprocessor (Preprocessor): preprocessor
        """

        self.dataset = pd.read_csv(file_path, sep='\t')[:1000]
        self.preprocessor = preprocessor
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, text, label = self.dataset.iloc[idx].tolist()

        input_id = self.preprocessor.get_input_id(text)
        attention_mask = self.preprocessor.get_attention_mask(input_id)

        input_id = torch.tensor(input_id)
        attention_mask = torch.tensor(attention_mask)
        label = torch.tensor(label)
       
        return input_id, attention_mask, label
