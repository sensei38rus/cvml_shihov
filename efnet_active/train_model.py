import torch
from torchvision import datasets, transforms
import cv2
from torch import nn, optim
from pathlib import Path
import torchvision
import numpy as np
from PIL import Image
import time
from collections import deque

save_path = Path(__file__).parent
model_path = save_path/"model.pth"

def build_model():
    # Загружаем предобученную EfficientNet-B0
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    
    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False
    
    # Получаем количество входных признаков для классификатора
    # У EfficientNet-B0 классификатор находится в model.classifier
    in_features = model.classifier[1].in_features
    
    # Заменяем последний слой на бинарную классификацию
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    
    # Размораживаем только новый слой
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print("Модель загружена из сохраненного файла")
    
    return model

model = build_model()


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Трансформации для EfficientNet-B0 (нужен размер 224x224)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train(buffer):
    if len(buffer) < 10:
        return None
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze(-1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted = model(tensor).squeeze(-1)
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no person"
    return label, prob

class Buffer():
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)
    
    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels

cap = cv2.VideoCapture(0)
buffer = Buffer()
count_labeled = 0

print("Управление:")
print("1 - отметить как 'person'")
print("2 - отметить как 'no person'")
print("p - предсказание")
print("s - сохранить модель")
print("q - выход")

while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if key == ord("q"):
        break
    elif key == ord("1"):  # person
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
        print(f"Добавлен образец person ({len(buffer)}/{buffer.frames.maxlen})")
    elif key == ord("2"):  # no person
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
        print(f"Добавлен образец no person ({len(buffer)}/{buffer.frames.maxlen})")
    elif key == ord("p"):
        t = time.perf_counter()
        label, confidence = predict(frame)
        inference_time = time.perf_counter() - t
        print(f"Время инференса: {inference_time:.3f}с")
        print(f"Предсказание: {label} (уверенность: {confidence:.3f})")
    elif key == ord("s"):  # save model
        torch.save(model.state_dict(), save_path / "model.pth")
        print("Модель сохранена")

    if count_labeled >= buffer.frames.maxlen:
        loss = train(buffer)
        if loss:
            print(f"Loss = {loss:.4f}")
        count_labeled = 0