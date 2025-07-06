import asyncio

from aiogram import Bot, Dispatcher
from bot.app.handlers import router


async def main():
    bot = Bot(token='YOUR_TG_TOKEN')
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот выключен')