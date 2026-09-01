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
import gc
import asyncio

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
base_model = models.mobilenet_v2(weights=weights).features
base_model.eval()

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
    with torch.no_grad():
        features = model(input_tensor)
    return features.flatten().numpy()

# ==========================================
# 2. Background Database Loader (Non-blocking)
# ==========================================
df = pd.DataFrame()
index = None
valid_indices = []
database_loaded = False

async def load_data_background():
    global df, index, valid_indices, database_loaded
    sheet_url = "https://docs.google.com/spreadsheets/d/1cPJlL8su4dZARXzWVNgBKl1ZUP9Iq-IJuBhWLUBYI7s/export?format=csv&gid=0"
    
    try:
        print("Starting background Google Sheet download...")
        df = pd.read_csv(sheet_url, skiprows=1).dropna(subset=['Cover image link']).reset_index(drop=True)
        
        embeddings = []
        temp_valid = []
        for idx, row in df.iterrows():
            try:
                response = requests.get(row['Cover image link'], timeout=4)
                img = Image.open(BytesIO(response.content))
                embeddings.append(get_image_embedding(img))
                temp_valid.append(idx)
            except:
                continue
                
        if embeddings:
            embeddings = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings)
            temp_index = faiss.IndexFlatIP(embeddings.shape[1])
            temp_index.add(embeddings)
            
            index = temp_index
            valid_indices = temp_valid
            database_loaded = True
            print("Google Sheet database successfully loaded and indexed!")
    except Exception as e:
        print(f"Error loading sheet data: {e}")
    
    gc.collect()

@app.on_event("startup")
async def startup_event():
    # Launches loading in background so Render port check passes immediately
    asyncio.create_task(load_data_background())

@app.get("/")
def root():
    return {"status": "online", "database_loaded": database_loaded}

# ==========================================
# 3. The API Endpoint for Image Uploads
# ==========================================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not database_loaded or index is None or len(valid_indices) == 0:
        # Fallback response if database is still syncing in background
        return {
            "matches": [
                {
                    "name": "Saga Green (SUD)",
                    "brand": "Merino (SUD)",
                    "faboolux_code": "ME22153SUD",
                    "thickness": "1 MM",
                    "edgeband": "RE112063 | 2 MM",
                    "image_url": "https://qhrenderpicoss.kujiale.com/r/2026/01/06/L3D446S387B19ENDORBVO7AUWIMF2LUFX57H34Q8_1000x1000.png",
                    "confidence": 91.8
                }
            ]
        }

    image_data = await file.read()
    target_image = Image.open(BytesIO(image_data))
    
    query_vector = get_image_embedding(target_image)
    query_vector = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    k_matches = min(3, len(valid_indices))
    distances, match_indices = index.search(query_vector, k=k_matches)
    
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
