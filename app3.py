import streamlit as st
st.set_page_config(page_title="Retinal Fundus Analysis", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    font-size: 18px !important;
    font-weight: 600;
}

h1 {
    font-size: 42px !important;
}
h2 {
    font-size: 32px !important;
}
h3 {
    font-size: 26px !important;
}
</style>
""", unsafe_allow_html=True)

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CLASS NAMES =================
class_names = [
    'DRY AMD',
    'Healthy',
    'MILDNPDR',
    'MODERATE',
    'SEVERNPDR',
    'WET AMD',
    'mild-gla',
    'moderate_gla',
    'sever_gla'
]

# ================= MODEL =================
model = models.efficientnet_b0(pretrained=False)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 9)
    )
)

model.load_state_dict(torch.load("C:/EyeProject/model/best_combined_model.pth", map_location=device))
model.to(device)
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ================= TITLE =================
st.markdown("<h1 style='text-align:center;'>Retinal Fundus Analysis</h1>", unsafe_allow_html=True)

# ================= INPUT =================
st.subheader("🧑 Patient Details")

c1, c2, c3 = st.columns(3)
with c1:
    name = st.text_input("Name")
with c2:
    age = st.number_input("Age", 1, 120)
with c3:
    gender = st.selectbox("Gender", ["Male","Female","Other"])

st.subheader("🏥 Medical History")

d1, d2, d3 = st.columns(3)
with d1:
    diabetes = st.checkbox("Diabetes")
with d2:
    bp = st.checkbox("High BP")
with d3:
    heart = st.checkbox("Heart Disease")

other_disease = st.text_input("Other Disease (optional)")

uploaded_file = st.file_uploader("📤 Upload Fundus Image", type=["jpg","png","jpeg"])

# ================= GRADCAM =================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        output = self.model(x)
        pred_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, pred_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1,2))
        cam = torch.zeros(activations.shape[1:], device=device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

# ================= PDF =================
def create_pdf(name, age, gender, disease, severity, confidence):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Retinal Fundus Analysis Report", styles["Title"]))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))

    content.append(Spacer(1,10))
    content.append(Paragraph(f"Prediction: {disease}", styles["Normal"]))
    content.append(Paragraph(f"Severity: {severity}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence}%", styles["Normal"]))

    content.append(Spacer(1,10))
    content.append(Paragraph("⚠️ Consult an ophthalmologist for confirmation.", styles["Normal"]))

    doc.build(content)
    return temp_file.name

# ================= MAIN =================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    confidence, pred = torch.max(probs, 0)
    label = class_names[pred.item()]
    confidence = round(confidence.item()*100,2)

    # ================= TYPE =================
    label_lower = label.lower()

    if "amd" in label_lower:
        disease = "Age-related Macular Degeneration"
    elif "gla" in label_lower:
        disease = "Glaucoma"
    elif "npdr" in label_lower:
        disease = "Diabetic Retinopathy"
    elif "healthy" in label_lower:
        disease = "Healthy"
    else:
        disease = "Unknown"

    # ================= SEVERITY =================
    if "mild" in label_lower:
        severity = "Mild"
    elif "moderate" in label_lower:
        severity = "Moderate"
    elif "sever" in label_lower or "wet" in label_lower:
        severity = "Severe"
    elif "dry" in label_lower:
        severity = "Early"
    else:
        severity = "None"

    # ================= RESULT =================
    st.markdown("## 🔍 Prediction Result")
    st.success(f"Detected: {disease}")
    st.write(f"Confidence: **{confidence}%**")

    if disease != "Healthy":
        st.warning(f"Severity Level: **{severity}**")
    else:
        st.info("Eye appears healthy")

    # ================= RECOMMENDATIONS =================
    st.markdown("### 🩺 Recommendations")

    if disease == "Healthy":
        st.success("Maintain healthy lifestyle.")

    elif "amd" in label_lower:
        if "dry" in label_lower:
            st.write("• Healthy diet, leafy greens, fish, antioxidants.")
        elif "wet" in label_lower:
            st.error("• Immediate treatment required!")

    else:
        if severity == "Mild":
            st.write("• Regular monitoring.")
        elif severity == "Moderate":
            st.write("• Visit doctor soon.")
        elif severity == "Severe":
            st.error("• Immediate attention required!")

    # ================= GRADCAM =================
    target_layer = None
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            target_layer = layer
            break

    cam = GradCAM(model, target_layer).generate(img_tensor)

    cam = cv2.resize(cam, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        np.array(image.resize((224,224))),
        0.6,
        heatmap,
        0.4,
        0
    )

    # ================= SIDE BY SIDE =================
    st.markdown("### 🖼️ Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, width=350)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        st.image(overlay, width=350)

    # ================= PDF =================
    pdf_path = create_pdf(name, age, gender, disease, severity, confidence)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Report", f, file_name="report.pdf")

    # ================= DISCLAIMER =================
    st.markdown("""
    ---
    ⚠️ **Disclaimer:** This is an AI-based system.  
    Always consult a qualified ophthalmologist before making medical decisions.
    """)