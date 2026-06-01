import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.draw import disk, circle_perimeter, rectangle, rectangle_perimeter, polygon, polygon_perimeter
from pathlib import Path

path = Path(__file__).parent
# ==========================================
# 1. Генерация датасета (skimage.draw + PyTorch Dataset)
# ==========================================
class ShapeDataset(Dataset):
    def __init__(self, num_samples, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        shape_type = random.choice(['circle', 'square', 'triangle'])

        try:
            if shape_type == 'circle':
                r = random.randint(15, self.img_size // 3)
                r_c = random.randint(r + 5, self.img_size - r - 5)
                c_c = random.randint(r + 5, self.img_size - r - 5)
                
                rr, cc = disk((r_c, c_c), r, shape=img.shape)
                rr_p, cc_p = circle_perimeter(r_c, c_c, r, shape=mask.shape)

            elif shape_type == 'square':
                w = random.randint(20, self.img_size // 2)
                h = w # Квадрат
                start_r = random.randint(5, self.img_size - h - 5)
                start_c = random.randint(5, self.img_size - w - 5)
                
                # Координаты для skimage (start, extent)
                rr, cc = rectangle((start_r, start_c), extent=(h, w), shape=img.shape)
                rr_p, cc_p = rectangle_perimeter((start_r, start_c), extent=(h, w), shape=mask.shape)

            elif shape_type == 'triangle':
                # Генерируем 3 случайные точки
                r = np.random.randint(10, self.img_size - 10, 3)
                c = np.random.randint(10, self.img_size - 10, 3)
                
                rr, cc = polygon(r, c, shape=img.shape)
                rr_p, cc_p = polygon_perimeter(r, c, shape=mask.shape)
                
            img[rr, cc] = 1.0
            mask[rr_p, cc_p] = 1.0
        except ValueError:
            # Защита от выхода за границы при случайной генерации
            pass

        # Возвращаем тензоры формата [Channels, Height, Width]
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

# ==========================================
# 2. Архитектура модели (Упрощенная сегментационная CNN)
# ==========================================
class BoundaryNet(nn.Module):
    def __init__(self):
        super(BoundaryNet, self).__init__()
        # Энкодер (сжатие признаков)
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        
        # Декодер (восстановление разрешения)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()) # Выход от 0 до 1

    def forward(self, x):
        e1 = self.enc1(x)
        x = self.pool(e1)
        x = self.enc2(x)
        
        x = self.up(x)
        x = self.dec1(x)
        x = self.out(x)
        return x

# ==========================================
# 3. Логика обучения и сохранения
# ==========================================
# ==========================================
# 3. Логика обучения и сохранения
# ==========================================
def train_or_load_model(model_path=path/"model.pth", img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryNet().to(device)

    # Проверка наличия сохраненной модели
    if os.path.exists(model_path):
        print(f"[*] Найдена сохраненная модель '{model_path}'. Загрузка весов...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device

    print("[*] Модель не найдена. Начинаем генерацию датасета и обучение...")
    dataset = ShapeDataset(num_samples=2000, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # --- Подсчет Accuracy ---
            with torch.no_grad():
                # Бинаризуем предсказания (все что > 0.5 считаем за 1, остальное за 0)
                preds = (outputs > 0.5).float()
                # Считаем количество совпавших пикселей
                epoch_correct += (preds == masks).sum().item()
                # Считаем общее количество пикселей в батче
                epoch_total += masks.numel()

        # Вычисляем средние значения за эпоху
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_correct / epoch_total
        
        print(f"Эпоха {epoch+1}/{epochs}, Потеря: {avg_loss:.4f}, Точность: {avg_acc:.4f}")

    # Сохранение весов
    torch.save(model.state_dict(), model_path)
    print(f"[*] Обучение завершено. Модель сохранена как '{model_path}'.")
    model.eval()
    return model, device

# ==========================================
# 4. Интерфейс рисования (OpenCV)
# ==========================================
drawing = False
erasing = False
brush_size = 4
IMG_SIZE = 128
canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
pred_mask = None # Новая глобальная переменная для хранения предсказанных границ

def draw_callback(event, x, y, flags, param):
    global drawing, erasing, canvas, pred_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pred_mask = None # Сбрасываем предсказание при новом рисовании
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        pred_mask = None # Сбрасываем предсказание при стирании
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), brush_size, 1.0, -1)
        elif erasing:
            cv2.circle(canvas, (x, y), brush_size * 2, 0.0, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False

def main():
    global canvas, pred_mask
    
    # 1. Загрузка или обучение модели
    model, device = train_or_load_model(img_size=IMG_SIZE)
    
    # 2. Настройка окна OpenCV
    window_name = "Draw Shape (L-Click: Draw, R-Click: Erase, Enter: Predict, C: Clear, Q: Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 400)
    cv2.setMouseCallback(window_name, draw_callback)

    print("\nИнструкция:")
    print(" - ЛЕВАЯ кнопка мыши: Рисовать фигуру")
    print(" - ПРАВАЯ кнопка мыши: Стирать (ластик)")
    print(" - ENTER: Найти границы фигуры")
    print(" - 'C': Очистить холст")
    print(" - 'Q': Выход")

    while True:
        # Подготавливаем цветной фон для отображения
        display_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        
        # Если есть предсказание, накладываем его красным цветом поверх рисунка
        if pred_mask is not None:
            # Формат BGR: синий=0, зеленый=0, красный=1.0 (т.к. тип данных float32)
            display_img[pred_mask == 1.0] = [0, 0, 1.0] 

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): # Выход
            break
        elif key == ord('c'): # Очистка холста
            canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            pred_mask = None
        elif key == 13: # Enter -> Предикт
            input_tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(input_tensor)
                
            out = out.squeeze().cpu().numpy()
            pred_mask = (out > 0.5).astype(np.float32)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()