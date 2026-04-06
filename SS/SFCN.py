import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1 生成模拟数据集
def create_synthetic_data(num_samples=500, img_size=64, num_classes=3):
    images = []
    masks = []

    for _ in range(num_samples):
        # 随机生成图像（3通道）
        img = np.random.rand(img_size, img_size, 3).astype(np.float32)

        # 生成掩码（多类别）
        mask = np.zeros((img_size, img_size), dtype=np.int64)

        center_x, center_y = np.random.randint(20, img_size - 20, 2)
        radius = np.random.randint(8, 16)

        # 创建圆形区域（类别1）
        y, x = np.ogrid[:img_size, :img_size]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[circle_mask] = 1

        # 创建方形区域（类别2）
        x1, y1 = center_x + 15, center_y - 10
        x2, y2 = x1 + 20, y1 + 20
        square_mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        mask[square_mask] = 2

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# 2 自定义DataSet数据集
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.masks = torch.LongTensor(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# 3 FCN模型
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3):
        super(SimpleFCN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # middle layer
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# 训练函数
def train(model, train_loader, val_loader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

# 主程序
def main():
    # 参数配置
    IMG_SIZE = 64
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 30

    # 创建数据集
    print("生成模拟数据...")
    images, masks = create_synthetic_data(num_samples=800, img_size=IMG_SIZE, num_classes=NUM_CLASSES)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # 创建DataLoader
    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    model = SimpleFCN(num_classes=NUM_CLASSES, in_channels=3)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train(model, train_loader, val_loader, epochs=EPOCHS)


if __name__ == "__main__":
    main()

