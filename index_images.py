import os
import faiss
import pickle
import numpy as np
import torch
from PIL import Image
from rembg import remove
from transformers import CLIPProcessor, CLIPModel

# ➤ Qurilmani aniqlaymiz (GPU bo‘lsa ishlatamiz)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ➤ CLIP modelini yuklash
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ➤ Rasmdan fonni olib tashlash
def remove_background(image_path):
    img = Image.open(image_path)
    img_no_bg = remove(img)  # Fonni olib tashlaymiz
    return img_no_bg

# ➤ CLIP bilan embedding yaratish
def get_clip_embedding(image_path):
    img_no_bg = remove_background(image_path)  # Fonni olib tashlaymiz
    inputs = processor(images=img_no_bg, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).cpu().numpy().flatten()

    return embedding

# ➤ Rasmlar joylashgan papka
IMAGE_DIR = "products/"

# ➤ Barcha rasm fayllarini yig‘amiz
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

# ➤ CLIP modelidan foydalanib embedding yaratamiz
embeddings = np.array([get_clip_embedding(img) for img in image_paths])

# ➤ Faiss indeksi yaratamiz
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ➤ Indeks va rasm yo‘llarini saqlash
with open("image_index.pkl", "wb") as f:
    pickle.dump((index, image_paths), f)

print("✅ Indeks yaratildi va saqlandi!")
