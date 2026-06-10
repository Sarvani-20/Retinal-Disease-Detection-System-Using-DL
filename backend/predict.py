import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model('efficientnet_b0', pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 9)

model.load_state_dict(torch.load("C:\EyeProject\model\best_combined_model.pth"", map_location=device))
model.to(device)
model.eval()

class_names = ['Healthy','MILDNPDR','MODERATE','NODR','PDR','SEVERE NPDR','mild_gla','moderate_gla','severe_gla']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return "Invalid Image", 0

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return class_names[pred.item()], round(confidence.item()*100, 2)