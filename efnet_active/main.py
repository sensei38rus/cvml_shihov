from train_model import build_model, predict, transform
from pathlib import Path
import torch
import cv2

if __name__ == '__main__':
    path = Path(__file__).parent
    model_path = path / 'model.pth'
    
    if not model_path.exists():
        print("Model doesn't exist. Please run train_model.py first to train the model.")
        exit(1)
    
    # Загружаем модель
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("Режим ПРЕДСКАЗАНИЯ")
    print("Управление:")
    print("p - предсказание")
    print("q - выход")
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
    
    curr_label = ''
    curr_conf = 0.0
    
    while True:
        _, frame = cap.read()
        frame_copy = frame.copy()
        
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if key == ord('q'):
            break
        elif key == ord('p'):  # predict
            curr_label, curr_conf = predict(model, frame)
            print(f"Предсказание: {curr_label} (уверенность: {curr_conf:.3f})")
        
        # Отображаем результат на кадре
        if curr_label:
            cv2.putText(frame_copy, f"{curr_label} {curr_conf:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Camera", frame_copy)
    
    cap.release()
    cv2.destroyAllWindows()