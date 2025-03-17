import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import asyncio
import pickle
import torch
import faiss
import numpy as np
from PIL import Image
from rembg import remove
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType, InputFile
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from transformers import CLIPProcessor, CLIPModel
from aiogram.types import FSInputFile


# ➤ Bot tokenini yozing
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")

# ➤ Qurilma aniqlash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ➤ CLIP modelini yuklash
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ➤ Faiss indeks va rasm yo‘llarini yuklash
with open("image_index.pkl", "rb") as f:
    index, image_paths = pickle.load(f)

# ➤ Aiogram 3.x bo‘yicha Dispatcher va bot yaratish
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ➤ Fonni olib tashlash funksiyasi
def remove_background(image_path):
    img = Image.open(image_path)
    img_no_bg = remove(img)  # Fonni olib tashlaymiz
    return img_no_bg

# ➤ Rasm embeddingini olish
def get_clip_embedding(image_path):
    img_no_bg = remove_background(image_path)  # Fonni olib tashlaymiz
    inputs = processor(images=img_no_bg, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).cpu().numpy().flatten()

    return embedding

# ➤ Eng o‘xshash rasmni topish
def find_best_match(image_path):
    query_embedding = get_clip_embedding(image_path).reshape(1, -1)
    D, I = index.search(query_embedding, k=1)  # Eng yaqin 1 ta natija
    return image_paths[I[0][0]], D[0][0]

def find_best_matches(image_path, top_k=3):
    query_embedding = get_clip_embedding(image_path).reshape(1, -1)
    D, I = index.search(query_embedding, k=top_k)  # Eng yaqin `top_k` ta natija
    return [(image_paths[I[0][i]], D[0][i]) for i in range(top_k)]

# ➤ Bot /start buyrug‘iga javob beradi
@dp.message(CommandStart())
async def start_command(message: Message):
    await message.answer("👋 Assalomu alaykum! Menga rasm yuboring va men unga o‘xshash mahsulotni topib beraman.")

# ➤ Foydalanuvchi rasm yuborganda ishlaydigan funksiya
@dp.message(lambda message: message.photo)
async def handle_photo(message: Message, state: FSMContext):
    photo = message.photo[-1]
    photo_path = f"user_images/{message.from_user.id}.jpg"
    os.makedirs("user_images", exist_ok=True)
    await bot.download(photo, destination=photo_path)

    matches = find_best_matches(photo_path, top_k=3)

    found = False
    for match, distance in matches:
        if distance < 100:  # Masofa past bo‘lsa, o‘xshash mahsulot deb hisoblaymiz
            await message.answer_photo(photo=FSInputFile(match), caption="🔍 Eng o‘xshash mahsulot!")
            found = True

    if not found:
        await message.answer("❌ Ushbu mahsulot do‘konda topilmadi.")

# ➤ Botni ishga tushirish
async def main():
    print("🤖 Bot ishga tushdi!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
