import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ====== 配置路径 ======
model_path = r"G:\FER\emotion_resnet18_best.pth"
face_image_path = r"G:\videochat\my_design\CNCLIP_keyframes_test_2\faces\comp_kf_01790_face3.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 标签定义（FER七类）======
emotion_labels = ['愤怒 Angry', '厌恶 Disgust', '恐惧 Fear', '高兴 Happy', '悲伤 Sad', '惊讶 Surprise', '中性 Neutral']

# ====== 图像预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== 加载模型 ======
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(emotion_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ====== 情绪预测函数 ======
def predict_emotion(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return emotion_labels[predicted]

# ====== 主程序 ======
if __name__ == "__main__":
    model = load_model()
    emotion = predict_emotion(face_image_path, model)
    print(f"😊 预测结果：{emotion}")
