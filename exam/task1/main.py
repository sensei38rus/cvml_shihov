import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
from skimage import draw
import random
import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

save_path = Path(__file__).parent
MODEL_PATH = save_path/"model.pth"
CLASSES = ['Circle', 'Square', 'Triangle']
IMG_SIZE = 64

class ShapeDataset(Dataset):
    def __init__(self, num_samples = 3000, img_size = 64):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size

        # массив изображений 
        self.X = np.zeros((num_samples, 1, img_size, img_size), dtype = np.float32)
        # массив меток - 0 круг, 1 - квадрат, 2 - треугольник
        self.y = np.zeros(num_samples, dtype = np.int64)

        for i in range(num_samples):
            shape_type = random.randint(0,2) 
            self.y[i] = shape_type
            # пустой массив для холста
            img = np.zeros((img_size, img_size), dtype=np.float32)
            # размер фигуры
            size = random.randint(10,25)
            center_x = random.randint(size + 5, img_size - size - 5)
            center_y = random.randint(size + 5, img_size - size - 5)

            if shape_type == 0: # круг
                rr, cc = draw.disk((center_y, center_x),size,shape=img.shape)
                img[rr,cc] = 1.0
            
            elif shape_type == 1: # квадрат
                #4 угла для квадрата
                r = [center_y - size, center_y - size, center_y + size, center_y + size]
                c = [center_x - size, center_x + size, center_x + size, center_x - size]
                rr, cc = draw.polygon(r,c,shape=img.shape)
                img[rr,cc] = 1.0
            
            elif shape_type == 2: 
                r = [center_y-size, center_y + size, center_y + size]
                c = [center_x, center_x - size, center_x + size]
                rr, cc = draw.polygon(r,c,shape=img.shape)
                img[rr,cc] = 1.0

            # шум
            noise = np.random.normal(0,0.05, img.shape)
            # обрезаем в допустимом диапозоне
            img = np.clip(img+noise, 0, 1) 
            self.X[i, 0] = img

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    

class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128,3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_model(model, dataset, epochs = 5):
    print("Начало обучения модели")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs,labels in loader:
            optimizer.zero_grad() # обнуляем градиенты
            preds = model(inputs)
            loss = criterion(preds,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Эпоха {epoch+1}/{epochs}, Потеря: {running_loss/len(loader):.4f}")
    print("Обучение завершено!")

drawing = False
def draw_shape(event, x, y, flags, param):
    global drawing 
    canvas = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(canvas, (x,y), 8, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(canvas, (x, y), 8, 255, -1)
    
def run(model):
    print("\nОткрытие окна для рисования...")
    print("Управление:")
    print("  - Рисуйте левой кнопкой мыши.")
    print("  - Нажмите 'c' (англ) для очистки холста.")
    print("  - Нажмите 'q' (англ) для выхода.")

    model.eval()
    canvas_size = 400
    canvas = np.zeros((canvas_size,canvas_size), dtype=np.uint8)

    cv2.namedWindow("Shape Classifier")
    cv2.setMouseCallback("Shape Classifier", draw_shape,canvas)

    while True:
        display_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        points = cv2.findNonZero(canvas)
        if points is not None:
            x,y,w,h = cv2.boundingRect(points)
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

            pad = 20
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(canvas_size, x + w + pad)
            y_max = min(canvas_size, y + h + pad)
            
            roi = canvas[y_min:y_max, x_min:x_max]

            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi_normalized = roi_resized.astype(np.float32) / 255.0
                input_tensor = torch.tensor(roi_normalized).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    preds = model(input_tensor)
                    probs = F.softmax(preds,dim=1)
                    prob, predicted = torch.max(probs,1)

                    class_name = CLASSES[predicted.item()]
                    confidence = prob.item() * 100
                    text = f"{class_name}: {confidence:.1f}%"
                    cv2.putText(display_img, text, (x, max(20, y - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Shape Classifier', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0

    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = ShapeClassifier()
    if os.path.exists(MODEL_PATH):
        print(f"Найдены сохраненные веса '{MODEL_PATH}'. Загрузка модели...")
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Модель успешно загружена и готова к работе!")
    else:
        train_dataset = ShapeDataset(num_samples=3000, img_size=IMG_SIZE)
        train_model(model, train_dataset, epochs=4)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Веса модели успешно сохранены в файл: {MODEL_PATH}")
    run(model)