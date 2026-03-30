from train_model import CyrillicCNN, CyrillicMNISTDataset  # Изменен импорт
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256

    path = Path(__file__).parent
    data_path = path / 'Cyrillic'
    model_path = path / 'model.pth'  
    save_path = path / 'pred.png'

    if not model_path.exists():
        print(f"Model doesn't exist at {model_path}. Please run train_model.py")
        return

    all_dataset = CyrillicMNISTDataset(data_path, is_train=False)

    _, test_samples = train_test_split(
        all_dataset.samples,
        test_size=0.2,
        random_state=42,
        stratify=[s[1] for s in all_dataset.samples]
    )

    test_dataset = CyrillicMNISTDataset(data_path, is_train=False)
    test_dataset.samples = test_samples
   
    num_workers = 8
    try:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True if num_workers > 0 else False
        )
    except RuntimeError as e:
        print(f"Error creating DataLoader with num_workers={num_workers}: {e}")
        print("Falling back to num_workers=0...")
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )

    model = CyrillicCNN().to(device)  # Используем улучшенную модель
    
    # Загрузка модели из чекпоинта
    checkpoint = torch.load(model_path, map_location=device)
    
  
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss at save: {checkpoint.get('val_loss', 'unknown'):.3f}")
        print(f"Validation accuracy at save: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
       
        model.load_state_dict(checkpoint)
        print("Loaded model in old format")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
       
        images_show, labels_show = next(iter(test_loader))
        images_show = images_show.to(device)
        outputs = model(images_show)
        _, pred_show = torch.max(outputs, 1)
        images_show = images_show.to('cpu')

        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()

        for i in range(16):
            axs[i].imshow(images_show[i].squeeze(), cmap='gray')
            label_class = test_dataset.ncls_to_label[labels_show[i].item()]
            label_class_pred = test_dataset.ncls_to_label[pred_show[i].item()]
            
            color = 'green' if label_class_pred == label_class else 'red'
            axs[i].set_title(f'pred={label_class_pred}\ntrue={label_class}', color=color, fontsize=10)
            axs[i].axis('off')

        plt.tight_layout()
        print(f"Saving to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path.exists()}")
        plt.show()

       
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f'\nTest accuracy on {total} samples: {acc:.2f}%')
    
    del test_loader

if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()