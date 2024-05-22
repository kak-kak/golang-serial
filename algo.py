import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

# 画像シーケンスデータセットの定義
class ImageSequenceDataset(Dataset):
    def __init__(self, sequences, sequence_length=5):
        self.sequences = sequences
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences) - self.sequence_length + 1

    def __getitem__(self, idx):
        seq = self.sequences[idx:idx + self.sequence_length]
        return torch.tensor(seq, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

# VAE-LSTMの定義
class VAE_LSTM(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), latent_dim=128, hidden_dim=256, sequence_length=5):
        super(VAE_LSTM, self).__init__()
        self.sequence_length = sequence_length
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_decode = nn.Linear(hidden_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        batch_size, sequence_length, c, h, w = x.size()
        x = x.view(batch_size * sequence_length, c, h, w)
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        batch_size, sequence_length, _ = z.size()
        z = z.view(batch_size, sequence_length, -1)
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out.contiguous().view(batch_size * sequence_length, -1)
        recon_x = self.decode(lstm_out)
        recon_x = recon_x.view(batch_size, sequence_length, 3, 128, 128)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae_lstm(model, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader.dataset)}')

def detect_anomalies(model, dataloader, threshold=0.01):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            if loss > threshold:
                anomalies.append(batch.cpu().numpy())
    return anomalies

# 画像シーケンスの準備（サンプルデータ）
sequences = []  # ここにNumPy形式の画像シーケンスデータを追加してください (例: sequences = np.load('sequences.npy'))
# 例: ダミーデータを生成
for _ in range(100):
    sequence = [np.random.rand(128, 128, 3) for _ in range(5)]
    sequences.append(sequence)
sequences = np.array(sequences)

# データセットとデータローダーの作成
sequence_length = 5
dataset = ImageSequenceDataset(sequences, sequence_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# VAE-LSTMの初期化とトレーニング
vae_lstm = VAE_LSTM().cuda()
train_vae_lstm(vae_lstm, dataloader, epochs=10)

# 異常検知
anomalies = detect_anomalies(vae_lstm, dataloader)
print(f"Detected {len(anomalies)} anomalies")

# 異常シーケンスの表示（例）
for anomaly in anomalies:
    for seq in anomaly:
        for img in seq:
            img = img.transpose(1, 2, 0) * 255.0
            img = img.astype(np.uint8)
            cv2.imshow("Anomaly", img)
            cv2.waitKey(0)
cv2.destroyAllWindows()
