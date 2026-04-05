import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 1 定义简单的UNet模型
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # 编码器（不包含pooling的单独层）
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # 中间层
        self.mid_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.out = nn.Conv2d(64, 2, 1)  # 2类:背景和目标

    def forward(self, x):
        # 编码路径（保存每层输出用于跳跃连接）
        e1 = self.enc1_conv(x)  # 64x64x64
        p1 = self.pool1(e1)  # 64x32x32

        e2 = self.enc2_conv(p1)  # 128x32x32
        p2 = self.pool2(e2)  # 128x16x16

        e3 = self.enc3_conv(p2)  # 256x16x16
        p3 = self.pool3(e3)  # 256x8x8

        # 中间层
        mid = self.mid_conv(p3)  # 512x8x8

        # 解码路径
        d3 = self.up3(mid)  # 256x16x16
        # 注意：e3的尺寸是16x16，需要裁剪d3到相同尺寸
        d3 = torch.cat([d3, e3], dim=1)  # 512x16x16
        d3 = self.dec3_conv(d3)  # 256x16x16

        d2 = self.up2(d3)  # 128x32x32
        d2 = torch.cat([d2, e2], dim=1)  # 256x32x32
        d2 = self.dec2_conv(d2)  # 128x32x32

        d1 = self.up1(d2)  # 64x64x64
        d1 = torch.cat([d1, e1], dim=1)  # 128x64x64
        d1 = self.dec1_conv(d1)  # 64x64x64

        out = self.out(d1)  # 2x64x64
        return out

# 2 合成数据集
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, size=64):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像
        img = np.random.randn(3, self.size, self.size)
        mask = np.zeros((self.size, self.size), dtype=np.int64)

        # 随机画一个圆形作为目标
        center_x = np.random.randint(20, self.size - 20)
        center_y = np.random.randint(20, self.size - 20)
        radius = np.random.randint(5, 15)

        y, x = np.ogrid[:self.size, :self.size]
        circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[circle] = 1

        # 给图像添加一些颜色信息
        img[0, circle] += 1.0  # 红色通道增强
        img[1, circle] += 0.5  # 绿色通道增强

        return torch.FloatTensor(img), torch.LongTensor(mask)

# 3 训练函数
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


# 4 可视化预测结果
def visualize_prediction(model, dataset, idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    img, true_mask = dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu()

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 归一化图像显示
    img_display = img.permute(1, 2, 0).numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    axes[0].imshow(img_display)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask.numpy(), cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask.numpy(), cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# 5. 计算IoU指标
def compute_iou(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    iou_sum = 0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            # 计算IoU (目标类，索引为1)
            intersection = ((preds == 1) & (masks == 1)).sum().float()
            union = ((preds == 1) | (masks == 1)).sum().float()

            if union > 0:
                iou = intersection / union
                iou_sum += iou
                num_batches += 1

    return (iou_sum / num_batches).item() if num_batches > 0 else 0

# 主程序
if __name__ == "__main__":
    # 创建数据集
    print("创建合成数据集...")
    dataset = SyntheticDataset(num_samples=1000, size=64)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 创建模型
    print("创建UNet模型...")
    model = SimpleUNet()

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=10)

    # 计算验证集IoU
    val_iou = compute_iou(model, val_loader)
    print(f"\n验证集IoU: {val_iou:.4f}")

    # 可视化结果
    print("\n可视化预测结果...")
    visualize_prediction(model, val_dataset, idx=0)

    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

