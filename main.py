import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

# 🔑 Token (TOKEN-ni .env fayldan olamiz)
import os
TOKEN = os.getenv("TOKEN")

# 🤖 Bot va Dispatcher yaratamiz
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("👋 Salom! Bot ishga tushdi!")

async def main():
    print("🤖 Bot ishlayapti...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
