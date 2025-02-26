import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model for plant disease detection (35 classes)
class CNN_plant_disease(torch.nn.Module):
    def __init__(self):
        super(CNN_plant_disease, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.maxpool3 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 26 * 26, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 35)  # Ensure 35 classes to match the saved model
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, X):
        X = self.relu(self.maxpool1(self.conv1(X)))
        X = self.relu(self.maxpool2(self.conv2(X)))
        X = self.relu(self.maxpool3(self.conv3(X)))
        X = X.view(X.size(0), -1)
        X = self.relu(self.fc1(X))
        X = self.dropout(self.relu(self.fc2(X)))
        X = self.fc3(X)
        return X

# Load the pre-trained model
model = CNN_plant_disease()
model.load_state_dict(torch.load('plant_disease_detection_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#Background-color
st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #3463ad;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app UI
st.markdown(
                """
                <style>
                .custom-title {
                    color: #0c0d0d;  /* Red color */
                    font-size: 38px;
                    font-family: 'playfair display';
                    text-align:center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

st.markdown(f'<h1 class="custom-title">Plant Disease Detection</h1>', unsafe_allow_html=True)

st.write("Upload a plant leaf image to detect the disease.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for the model
    image = image_transforms(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    
    # Load class labels (replace with the path to your class labels)
    class_names = os.listdir(r'C:\Users\Dell\OneDrive\Desktop\database\Plant_disease_detection\archive (6)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train')  # Adjust to where your class folders are stored
    
    # Show the predicted result

    st.markdown(
                """
                <style>
                .custom-title {
                    color: #0c0d0d;  /* Red color */
                    font-size: 30px;
                    font-family: 'playfair display';
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    st.markdown(f'<h1 class="custom-title">Prediction result: {class_names[prediction.item()]}</h1>', unsafe_allow_html=True)
    
