from pathlib import Path
import time

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CyrillicMNISTDataset(Dataset):
  def __init__(self, path, is_train=False):
    self.path = Path(path)
    self.samples = []
    self.ncls_to_label = {}

    for ncls, cls in enumerate(sorted(self.path.glob('*'))):
      label = str(cls.name)
      self.ncls_to_label[ncls] = label

      for image_path in sorted(cls.glob('*.png')):
        self.samples.append((image_path, ncls))

  
    train_transforms = transforms.Compose([
      transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.RandomAffine(
         degrees=(-15, 15),  
         translate=(0.1, 0.1), 
         scale=(0.85, 1.15) 
      ),
      transforms.RandomRotation(5),  
      transforms.ToTensor(),
    ])
    
    
    test_transforms = transforms.Compose([
      transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.ToTensor(),
    ])
    
    self.transforms = train_transforms if is_train else test_transforms

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    image_path, ncls = self.samples[idx]
    with Image.open(image_path) as img:
      mask = img.getchannel('A')
    mask = self.transforms(mask)
    mask = (mask > 0).float()
    mask = (mask - 0.5) / 0.5
      
    return mask, ncls

class CyrillicCNN(nn.Module):
  def __init__(self):
    super(CyrillicCNN, self).__init__()
    
   
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU(inplace=True) 
    self.pool1 = nn.MaxPool2d(2, 2) 
    
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(2, 2) 

    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.relu3 = nn.ReLU(inplace=True)
    self.pool3 = nn.MaxPool2d(2, 2)
    
    
    self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.relu4 = nn.ReLU(inplace=True)
    self.pool4 = nn.MaxPool2d(2, 2)
    
   
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(256 * 4 * 4, 512)  
    self.relu5 = nn.ReLU(inplace=True)
    self.dropout1 = nn.Dropout(0.4)  
    self.fc2 = nn.Linear(512, 256)
    self.relu6 = nn.ReLU(inplace=True)
    self.dropout2 = nn.Dropout(0.3)
    self.fc3 = nn.Linear(256, 34)
    
  
    self._initialize_weights()
    
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.pool3(x)
    
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.relu4(x)
    x = self.pool4(x)
    
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu5(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = self.relu6(x)
    x = self.dropout2(x)
    x = self.fc3(x)
    
    return x

if __name__ == '__main__':
  path = Path(__file__).parent
  data_path = path / 'Cyrillic'
  save_path = path
  model_path = save_path / "model.pth"

  all_dataset = CyrillicMNISTDataset(data_path)

  train_samples, test_samples = train_test_split(
    all_dataset.samples,
    test_size=0.2,
    random_state=42,
    stratify=[s[1] for s in all_dataset.samples]
  )

  train_dataset = CyrillicMNISTDataset(data_path, is_train=True)
  train_dataset.samples = train_samples

  test_dataset = CyrillicMNISTDataset(data_path, is_train=False)
  test_dataset.samples = test_samples

  
  batch_size = 256
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
 
  model = CyrillicCNN().to(device)
  
 
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")

  num_epochs = 100
  patience = 6  

  train_loss = []
  train_acc = []
  test_loss = []
  test_acc = []

  best_val_loss = float('inf')
  epochs_no_imporve = 0

  criterion = nn.CrossEntropyLoss()
 
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


  if not model_path.exists():
    for epoch in range(num_epochs):
      start = time.time()
      model.train()
      run_loss = 0.0
      total = 0
      correct = 0

      for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = (images.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        run_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      epoch_loss = run_loss / len(train_loader)
      epoch_acc = 100 * (correct / total)
      train_loss.append(epoch_loss)
      train_acc.append(epoch_acc)

      model.eval()
      val_loss, val_correct, val_total = 0.0, 0, 0

      with torch.no_grad():
        for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          val_loss += loss.item()
          
          _, predicted = torch.max(outputs.data, 1)
          val_total += labels.size(0)
          val_correct += (predicted == labels).sum().item()

      val_epoch_loss = val_loss / len(test_loader)
      val_epoch_acc = 100 * (val_correct / val_total)

      scheduler.step(val_epoch_loss)

      test_loss.append(val_epoch_loss)
      test_acc.append(val_epoch_acc)

     
      current_lr = optimizer.param_groups[0]['lr']
      
      if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_imporve = 0       
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'val_loss': val_epoch_loss,
          'val_acc': val_epoch_acc,
        }, model_path)
        print(f"Model saved with val_loss: {val_epoch_loss:.3f}, val_acc: {val_epoch_acc:.2f}%")
      else:
        epochs_no_imporve += 1
        if epochs_no_imporve >= patience:
          print(f"Early stopping triggered after {epoch} epochs")
          break
        
      print(f"Epoch {epoch}: Train Loss {epoch_loss:.3f}, Train Acc {epoch_acc:.2f}%")
      print(f"          Val Loss {val_epoch_loss:.3f}, Val Acc {val_epoch_acc:.2f}%")
      print(f"          LR: {current_lr:.6f}")

      end = time.time()
      print(f'time={end - start:.2f}s\n')

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(train_acc, label='Train acc')
    plt.plot(test_acc, label='Test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path / 'train.png')
    plt.show()

    print(f"Best validation loss: {best_val_loss:.3f}")
    print(f"Best validation accuracy: {max(test_acc):.2f}%")

    del train_loader
    del test_loader
  else:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.3f}, val_acc: {checkpoint['val_acc']:.2f}%")