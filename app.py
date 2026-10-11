import json
from pathlib import Path
from google import genai
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_weights.pth"
CLASS_PATH = BASE_DIR / "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to predict its disease "
    "using your trained ResNet-50 model."
)


# -----------------------------
# Load class names
# -----------------------------
@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, 38)
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Grad-CAM
# -----------------------------
def create_gradcam(model, input_tensor, class_index):
    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )
    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        logits = model(input_tensor)
        score = logits[0, class_index]
        score.backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        weights = gradient.mean(
            dim=(2, 3), keepdim=True
        )

        cam = (weights * activation).sum(dim=1)
        cam = F.relu(cam)[0]

        cam = cam.cpu().numpy()
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    finally:
        forward_handle.remove()
        backward_handle.remove()


# -----------------------------
# Upload and predict
# -----------------------------
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "Model files not found. Ensure best_resnet50.pth "
        "and class_names.json are beside app.py."
    )
    st.stop()

try:
    class_names = load_class_names()
    model = load_model()

    if len(class_names) != 38:
        st.error(
            f"Expected 38 classes, found {len(class_names)}."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded leaf image",
        width=350
    )

    if st.button("Predict Disease", type="primary"):
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        try:
            with st.spinner("Analyzing leaf image..."):
                input_tensor.requires_grad_(True)

                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)

                confidence, predicted_index = (
                    probabilities.max(dim=1)
                )

                predicted_index = predicted_index.item()
                confidence = confidence.item()

                heatmap = create_gradcam(
                    model, input_tensor, predicted_index
                )

                st.session_state["prediction"] = (
                    class_names[predicted_index]
                )
                st.session_state["confidence"] = confidence
                st.session_state["heatmap"] = heatmap
                st.session_state["image"] = image

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if "prediction" in st.session_state:
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted class",
            st.session_state["prediction"]
        )
        st.metric(
            "Model confidence",
            f"{st.session_state['confidence'] * 100:.2f}%"
        )

    with col2:
        original = np.array(st.session_state["image"])
        heatmap = st.session_state["heatmap"]

        fig, ax = plt.subplots()
        ax.imshow(original)
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.4,
            extent=[
                0, original.shape[1],
                original.shape[0], 0
            ],
            interpolation="bilinear"
        )
        ax.axis("off")
        ax.set_title("Grad-CAM Visualization")

        st.pyplot(fig)
        plt.close(fig)
# -----------------------------
# Gemini AI Explanation
# -----------------------------
st.subheader("🌱 AI Disease Explanation")

if "prediction" in st.session_state:
    if st.button("Generate Disease Explanation"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            disease_name = st.session_state["prediction"]
            confidence = st.session_state["confidence"]

            prompt = f"""
            You are an agricultural plant disease assistant.

            A ResNet-50 model predicted: {disease_name}
            Model confidence: {confidence * 100:.2f}%

            Explain this prediction in simple English.
            Include:
            1. What the disease is.
            2. Common symptoms.
            3. Possible causes.
            4. General management and prevention steps.

            Clearly state that this is an AI prediction and
            the disease should be confirmed by an expert.
            Do not claim that the prediction is certain.
            """

            with st.spinner("Generating explanation..."):
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

            st.markdown(response.text)

        except Exception as e:
            st.error(
                "Could not generate the explanation. "
                "Check your API key, model availability, "
                "internet connection, and API quota."
            )
            st.caption(str(e))