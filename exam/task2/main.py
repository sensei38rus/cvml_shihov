import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
path = Path(__file__).parent
# --- 1. Настройка параметров ---
CSV_PATH = path/'chinese/chinese_mnist.csv'
IMG_DIR = path/'chinese/data/data'
MODEL_PATH = path/'model.pth'
BATCH_SIZE = 64
EPOCHS = 30
VAL_SPLIT = 0.2
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Кастомный Dataset ---
class ChineseMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.data_frame['code'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        suite_id = row['suite_id']
        sample_id = row['sample_id']
        code = row['code']

        img_name = f"input_{suite_id}_{sample_id}_{code}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('L')
        label = self.class_to_idx[code]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. Архитектура Нейросети ---
class CNN(nn.Module):
    def __init__(self, num_classes=15):
        super(CNN, self).__init__()
        # Блок 1: 1 -> 32, 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Блок 2: 32 -> 64, 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Блок 3: 64 -> 128, 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- 4. Функция обучения одной эпохи ---
def run_epoch(model, dataloader, criterion, optimizer, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# --- 5. Функция обучения с early stopping ---
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, training=False)

        print(
            f"Эпоха [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping сработал на эпохе {epoch + 1}.")
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Лучшая модель сохранена в {MODEL_PATH}")

# --- 6. Интерактивная рисовалка на OpenCV ---
def draw_and_predict(model, transform, class_mapping):
    model.eval()
    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(canvas, (x, y), 5, (255), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), 5, (255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(canvas, (x, y), 5, (255), -1)

    cv2.namedWindow('Draw Character')
    cv2.setMouseCallback('Draw Character', draw)

    print("\n--- ИНСТРУКЦИЯ ---")
    print("Рисуйте иероглиф в открывшемся окне мышкой.")
    print("Нажмите 'Space' (Пробел) для распознавания.")
    print("Нажмите 'c' для очистки холста.")
    print("Нажмите 'q' или 'Esc' для выхода.")

    while True:
        cv2.imshow('Draw Character', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            canvas[:] = 0
        elif key == ord(' '):
            img_resized = cv2.resize(canvas, (64, 64))
            img_pil = Image.fromarray(img_resized)
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output.data, 1)
                pred_idx = predicted.item()

                original_code = [k for k, v in class_mapping.items() if v == pred_idx][0]
                print(f"-> Нейросеть думает, что это иероглиф с кодом: {original_code}")

        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

# --- Главная логика ---
if __name__ == '__main__':
    print(f"Используемое устройство: {DEVICE}")

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = ChineseMNISTDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=train_transform)
    num_classes = len(full_dataset.classes)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset = ChineseMNISTDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if os.path.exists(MODEL_PATH):
        print(f"Найдена сохраненная модель '{MODEL_PATH}'. Загрузка весов...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Сохраненная модель не найдена. Начинаем обучение...")
        train_model(model, train_loader, val_loader, criterion, optimizer)

    draw_and_predict(model, val_transform, full_dataset.class_to_idx)
