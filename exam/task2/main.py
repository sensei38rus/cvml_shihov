import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
path = Path(__file__).parent
# --- 1. Настройка параметров ---
CSV_PATH = path/'chinese/chinese_mnist.csv'
IMG_DIR = path/'chinese/data/data'
MODEL_PATH = path/'model.pth'
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Кастомный Dataset ---
class ChineseMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Уникальные коды для маппинга (в датасете они обычно 1-15)
        self.classes = sorted(self.data_frame['code'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Получаем данные из строки CSV
        row = self.data_frame.iloc[idx]
        suite_id = row['suite_id']
        sample_id = row['sample_id']
        code = row['code']

        # Формируем имя файла согласно структуре
        img_name = f"input_{suite_id}_{sample_id}_{code}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Загружаем изображение (в оттенках серого)
        image = Image.open(img_path).convert('L')
        label = self.class_to_idx[code]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. Архитектура Нейросети ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(SimpleCNN, self).__init__()
        # Изображения Chinese MNIST имеют размер 64x64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # -> 16 x 32 x 32
        x = self.pool(self.relu(self.conv2(x))) # -> 32 x 16 x 16
        x = x.view(-1, 32 * 16 * 16)            # Вытягиваем в вектор
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 4. Функция обучения ---
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    
                print(f"[Эпоха {epoch + 1}, Батч {i + 1}] Лосс: {running_loss / 100:.3f}")
                running_loss = 0.0
    
    print("Обучение завершено.")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

# --- 5. Интерактивная рисовалка на OpenCV ---
def draw_and_predict(model, transform, class_mapping):
    model.eval()
    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(canvas, (x, y), 10, (255), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), 10, (255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(canvas, (x, y), 10, (255), -1)

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
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        elif key == ord(' '):
            # Подготовка изображения для модели
            img_resized = cv2.resize(canvas, (64, 64))
            img_pil = Image.fromarray(img_resized)
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # Предсказание
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output.data, 1)
                pred_idx = predicted.item()
                
                # Ищем оригинальный код класса
                original_code = [k for k, v in class_mapping.items() if v == pred_idx][0]
                print(f"-> Нейросеть думает, что это иероглиф с кодом: {original_code}")

        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

def test_model_accuracy(model, test_loader):
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0
    
    # Отключаем расчет градиентов для ускорения и экономии памяти
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Точность модели на тестовой выборке ({total} изображений): {accuracy:.2f}%")

# --- Главная логика ---
if __name__ == '__main__':
    print(f"Используемое устройство: {DEVICE}")

    # Трансформации: в тензор и нормализация
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Инициализация датасета
    dataset = ChineseMNISTDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    # Инициализация модели
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Проверка на наличие сохраненной модели
    if os.path.exists(MODEL_PATH):
        print(f"Найдена сохраненная модель '{MODEL_PATH}'. Загрузка весов...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Сохраненная модель не найдена. Начинаем обучение...")
        train_model(model, train_loader, criterion, optimizer)
        
    test_model_accuracy(model, test_loader)
    # Запуск интерфейса рисования
    draw_and_predict(model, transform, dataset.class_to_idx)