import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import KFold

# 自作のデータセットクラス
class NPYDataset(Dataset):
    def __init__(self, file_list):
        self.data = [np.load(file) for file in file_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# ファイルリストの定義
file_list = ['file1.npy', 'file2.npy', 'file3.npy', 'file4.npy', 'file5.npy', 
             'file6.npy', 'file7.npy', 'file8.npy', 'file9.npy', 'file10.npy']

# データセットの作成
dataset = NPYDataset(file_list)

# クロスバリデーションのセットアップ
kf = KFold(n_splits=10)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SequentialSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=1, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=1, sampler=val_subsampler)
    
    # モデルの定義（例として簡単な線形モデル）
    model = torch.nn.Linear(dataset[0].shape[0], 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # トレーニング
    model.train()
    for epoch in range(10):  # エポック数は適宜変更してください
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)  # 適宜ターゲットの修正が必要です
            loss.backward()
            optimizer.step()

    # 検証
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            loss = criterion(outputs, batch)  # 適宜ターゲットの修正が必要です
            val_loss += loss.item()

    print(f'Fold {fold}, Validation Loss: {val_loss/len(val_loader)}')
