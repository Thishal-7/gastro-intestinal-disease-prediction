import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
import pickle
import random

from models import DenseNet121Model, InceptionResNetModel, Resnet50, EnsembleModel, load_lab_model
from utils import *

st.set_option('client.showErrorDetails', False)


# CONFIG

NUM_CLASSES = 27
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    "Accessory tools","Angiectasia","Barretts esophagus","Blood in lumen","Cecum",
    "Colon diverticula","Colon polyps","Colorectal cancer","Duodenal bulb",
    "Dyed-lifted-polyps","Dyed-resection-margins","Erythema","Esophageal varices",
    "Esophagitis","Gastric polyps","Gastroesophageal junction normal z-line",
    "Ileocecal valve","Mucosal inflammation large bowel","Normal esophagus",
    "Normal mucosa","Normal stomach","Pylorus",
    "Resected polyps","Resection margins","Retroflex rectum",
    "Small bowel terminal ileum","Ulcer"
]

st.set_page_config(page_title="Medical AI System", layout="wide")

# SIDEBAR

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🖼️ Image Prediction", "🧪 Lab Prediction"])

# LOAD MODELS

@st.cache_resource
def load_models():
    densenet = DenseNet121Model(NUM_CLASSES).to(DEVICE)
    inception = InceptionResNetModel(NUM_CLASSES).to(DEVICE)
    resnet50_model = Resnet50(NUM_CLASSES).to(DEVICE)

    map_location = torch.device('cpu')

    densenet.load_state_dict(torch.load('densenet121_best_model.pth', map_location=map_location))
    inception.load_state_dict(torch.load('InceptionResNet_model_best_model.pth', map_location=map_location))
    resnet50_model.load_state_dict(torch.load('Resnet50_best_model.pth', map_location=map_location))

    ensemble = EnsembleModel([densenet, inception, resnet50_model]).to(DEVICE)

    for model in [densenet, inception, resnet50_model, ensemble]:
        model.eval()

    return ensemble, resnet50_model



# IMAGE PREPROCESS

def process_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)



# DISEASE MAPPING

def map_prediction(label):

    mapping = {
        "Blood in stool": "⚠️ Gastrointestinal Bleeding / Possible Ulcer",
        "Abdominal pain": "🟡 Gastritis / Digestive Disorder",
        "Diarrhea": "🟠 Infection / IBS",
        "Constipation": "🟢 Functional Bowel Disorder",
        "Rectal bleeding": "🔴 Hemorrhoids / Colorectal Issue",
        "Bloating": "🟡 IBS / Indigestion",
        "Weight loss": "🔴 Chronic GI Disease / Malabsorption",
        "Appetite loss": "🟠 Gastritis / Infection",
        "Ulcer": "🔴 Peptic Ulcer Disease",
        "Colitis": "🟣 Inflammatory Bowel Disease (IBD)",
        "Gastritis": "🟡 Gastritis",
        "GERD": "🔵 Acid Reflux (GERD)",
        "IBS": "🟢 Irritable Bowel Syndrome",
    }

    return mapping.get(label, f"🩺 {label}")



# IMAGE PAGE

if page == "🖼️ Image Prediction":

    st.title("🔍 Medical Image Analysis with XAI")

    ensemble_model, base_model = load_models()

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            input_tensor = process_image(image).to(DEVICE)

    with col2:
        if uploaded_file:

            if st.button("Analyze Image"):

                with torch.no_grad():
                    output = ensemble_model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    top_p, top_class = torch.topk(probs, k=3)

                st.subheader("Predictions")

                for p, c in zip(top_p[0], top_class[0]):
                    st.write(f"{CLASS_NAMES[c.item()]} - {p.item()*100:.2f}%")

                # GradCAM
                st.subheader("Grad-CAM")

                target_layer = base_model.resnet.layer4[-1]

                fig_gradcam = plot_multiple_gradcams(
                    image=image,
                    models=[base_model],
                    model_names=['ResNet50'],
                    target_layers=[target_layer],
                    class_names=CLASS_NAMES,
                    device='cpu'
                )

                st.pyplot(fig_gradcam)

                # LIME
                st.subheader("LIME Explanation")

                with open('lime_explainer.pkl', 'rb') as f:
                    config = pickle.load(f)

                explainer = LimeModelExplainer(
                    model=base_model,
                    class_names=config['class_names'],
                    device='cpu'
                )

                img = image.resize((224, 224)).convert('RGB')
                img_np = np.array(img)

                explanation, predictions = explainer.explain_image(img_np)

                fig = plot_lime_results(explanation, predictions, img_np, CLASS_NAMES)

                st.pyplot(fig)



# LAB PAGE

elif page == "🧪 Lab Prediction":

    st.title("🧪 GI Disease Prediction")

    model, encoder, columns = load_lab_model()

    st.markdown("### Enter Key Patient Details")

    user_input = {
        "Age": st.number_input("Age", 0, 100, 25),
        "BMI": st.number_input("BMI", 10.0, 50.0, 22.0),
        "Family_History": st.selectbox("Family History", ["Yes", "No"]),
        "H_Pylori_Status": st.selectbox("H. Pylori", ["Yes", "No"]),
        "Smoking_Status": st.selectbox("Smoking", ["Yes", "No"]),
        "Alcohol_Use": st.number_input("Alcohol Use", 0.0, 50.0, 5.0),
        "Stress_Level": st.number_input("Stress Level", 0.0, 10.0, 3.0),
        "Abdominal_Pain": st.number_input("Abdominal Pain", 0.0, 10.0, 2.0),
    }

    col1, col2 = st.columns(2)

    # Predict
    with col1:
        if st.button("Predict Disease"):

            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=columns, fill_value=0)

            prediction = model.predict(input_df)
            predicted_label = encoder.inverse_transform(prediction)[0]

            final_output = map_prediction(predicted_label)

            st.subheader("🩺 Diagnosis Result")
            st.success(final_output)

    # Random
    with col2:
        if st.button("🎲 Random Test"):

            random_input = {
                "Age": random.randint(20, 70),
                "BMI": random.uniform(18, 35),
                "Family_History": random.choice(["Yes", "No"]),
                "H_Pylori_Status": random.choice(["Yes", "No"]),
                "Smoking_Status": random.choice(["Yes", "No"]),
                "Alcohol_Use": random.uniform(0, 20),
                "Stress_Level": random.uniform(0, 10),
                "Abdominal_Pain": random.uniform(0, 10),
            }

            input_df = pd.DataFrame([random_input])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=columns, fill_value=0)

            prediction = model.predict(input_df)
            predicted_label = encoder.inverse_transform(prediction)[0]

            final_output = map_prediction(predicted_label)

            st.subheader("🎲 Random Prediction")
            st.success(final_output)
            st.write(random_input)



# FOOTER

st.markdown("---")
st.markdown("### 🚀 AI Powered Medical Diagnosis System")
