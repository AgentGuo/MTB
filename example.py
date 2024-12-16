import mtb
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

# 创建训练集和测试集
train_dataset = mtb.MTDataset(features='svc1', split="train")
val_dataset = mtb.MTDataset(features='svc1', split="val")
test_dataset = mtb.MTDataset(features='svc1', split="test")

print(f'train_size: {len(train_dataset)}, val_size: {len(val_dataset)}, test_size: {len(test_dataset)}')
# 构建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 查看数据
for inputs, targets in train_loader:
    print("Inputs shape:", inputs.shape)  # (batch_size, seq_length, num_features)
    print("Targets shape:", targets.shape)  # (batch_size,)
    break

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 2, output_size = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 创建模型、损失函数和优化器
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs.squeeze(), targets).item()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# 测试模型，计算MAPE
model.eval()
test_loss = 0
mape = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs.squeeze(), targets).item()
        mape += torch.mean(torch.abs((targets - outputs.squeeze()) / targets)).item()

mape = mape / len(test_loader)
print(f"Test Loss: {test_loss / len(test_loader):.4f}, MAPE: {mape:.4f}")