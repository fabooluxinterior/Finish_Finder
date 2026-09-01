from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import faiss
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import uvicorn
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. Setup Lightweight AI Model (MobileNetV2)
# ==========================================
weights = models.MobileNet_V2_Weights.DEFAULT
# Load MobileNet's feature extractor
base_model = models.mobilenet_v2(weights=weights).features
base_model.eval()

# Add pooling to flatten the output into a 1D vector
pool = nn.AdaptiveAvgPool2d((1, 1))
model = nn.Sequential(base_model, pool)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image):
    image = image.convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad(): # <--- Fixed with an underscore
        features = model(input_tensor)
    return features.flatten().numpy()

# ==========================================
# 2. Load Data & Build Database on Startup
# ==========================================
df = pd.DataFrame()
index = None
valid_indices = []

@app.on_event("startup")
def load_data():
    global df, index, valid_indices
    sheet_url = "https://docs.google.com/spreadsheets/d/1cPJlL8su4dZARXzWVNgBKl1ZUP9Iq-IJuBhWLUBYI7s/export?format=csv&gid=0"
    
    df = pd.read_csv(sheet_url, skiprows=1).dropna(subset=['Cover image link']).reset_index(drop=True)
    
    embeddings = []
    for idx, row in df.iterrows():
        try:
            response = requests.get(row['Cover image link'], timeout=5)
            img = Image.open(BytesIO(response.content))
            embeddings.append(get_image_embedding(img))
            valid_indices.append(idx)
        except:
            continue
            
    if embeddings:
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    
    # Force memory cleanup after startup
    gc.collect()

# ==========================================
# 3. The API Endpoint for Image Uploads
# ==========================================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    target_image = Image.open(BytesIO(image_data))
    
    query_vector = get_image_embedding(target_image)
    query_vector = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    distances, match_indices = index.search(query_vector, k=3)
    
    results = []
    for rank, match_idx_in_faiss in enumerate(match_indices[0]):
        actual_df_idx = valid_indices[match_idx_in_faiss]
        match_data = df.iloc[actual_df_idx]
        
        results.append({
            "name": match_data.get('NAME', 'Unknown'),
            "brand": f"{match_data.get('BRAND', '')} ({match_data.get('BRAND FINISH', '')})",
            "faboolux_code": match_data.get('FABOOLUX CODE', 'N/A'),
            "thickness": match_data.get('THICKNESS', 'N/A'),
            "edgeband": f"{match_data.get('EDGEBAND CODE', '')} | {match_data.get('EDGEBAND THICKNESS', '')}",
            "image_url": match_data.get('Cover image link', ''),
            "confidence": float(distances[0][rank] * 100)
        })
        
    return {"matches": results}
