import logging
import re
from datetime import datetime
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

import bot.app.keyboards as kb
from pipeplines.ragPipeline_tgbot import DialogManager, prompt, prompt_paths

# Создаем папку logs, если она не существует
log_dir = Path("src/logs")
log_dir.mkdir(exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d")

# Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'tg_logs_{current_date}.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Роутер для обработки сообщений
router = Router()

# Инициализация менеджера диалогов
dialog_manager = DialogManager()

# Состояния бота
class TripPlanning(StatesGroup):
    user_request = State()
    waiting_for_response = State()
    waiting_for_rating = State()

def markdown_to_html(text: str) -> str:
    """Конвертация Markdown в HTML"""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'<code>\1</code>', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    return text

def log_user_action(user_id: int, username: str, action: str):
    """
    Логирует действия пользователя в едином формате
    :param user_id: ID пользователя Telegram
    :param username: username пользователя
    :param action: описание действия
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] User ID: {user_id}, Username: @{username}, Action: {action}"
    logger.info(log_message)

# ================= ОБРАБОТЧИКИ КОМАНД =================

@router.message(CommandStart())
async def cmd_start(message: Message, bot: Bot):
    """Обработчик команды /start"""
    user = message.from_user
    log_user_action(user.id, user.username, "started the bot")
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer('Привет!👋 \nЯ бот-помощник для планирования отдыха в Новосибирске и области. \n\nЕсли вы хотите узнать у меня о каком-либо месте или составить маршрут - выберите "Начать диалог". \nЕсли вы хотите узнать больше обо мне - нажмите "Возможности"', reply_markup=kb.main)

@router.message(F.text == 'Возможности')
async def cmd_help(message: Message, bot: Bot):
    """Обработчик кнопки 'Возможности'"""
    user = message.from_user
    log_user_action(user.id, user.username, "requested bot capabilities")
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer('✅ Что я умею? \n▪️Составлять маршруты по местам Новосибирской области с учетом продолжительности, бюджета и интересов пользователя (пример запроса: Составь маршрут на выходные по Академгородку) \n▪️Уточнять сведения о каком-либо месте для отдыха в НСО: его адрес, цену и описание (пример запроса: Где находится кинотеатр Победа?) \n▪️Поддерживать дружеский диалог \n\n   ℹ️ Я только-только учусь и у меня тоже есть слабые стороны:  \n▪️Я наврядли пойму, что такое "горыныч", но постараюсь вам помочь, если вы напишите: "ресторан Горыныч" \n▪️Могу повторяться и говорить о том, что не знаю больше никаких мест. Это ложь. Скорее всего, я знаю, но боюсь ошибиться. Задавайте, пожалуйста, мне вопросы с конкретными пожеланиями по локации и интересам (например: "где покушать грузинской кухни в Академгородке?") \n▪️К сожалению, не смогу помочь порекомендовать место "рядом" с чем-то, но могу попробовать найти что-то в указанном районе или улице\n\n  🆘 Если я буду нести чепуху, начните, пожалуйста, диалог заново.')

@router.message(F.text == 'Контакты')
async def contacts(message: Message, bot: Bot):
    """Обработчик кнопки 'Контакты'"""
    user = message.from_user
    log_user_action(user.id, user.username, "requested contacts")
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer('📩 По возникшим вопросам и предложениям можно обращаться к @alinshn₊˚⊹♡')

@router.message(F.text == 'Начать диалог снова')
async def restart_dialog(message: Message, state: FSMContext, bot: Bot):
    """Обработчик кнопки 'Начать диалог снова'"""
    user = message.from_user
    user_id = str(user.id)
    log_user_action(user.id, user.username, "restarted the dialog")
    dialog_manager.clear_history(user_id)
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer("🔄 Диалог сброшен", reply_markup=kb.main)
    await state.set_state(TripPlanning.user_request)

@router.message(F.text == 'Завершить диалог')
async def end_dialog(message: Message, state: FSMContext, bot: Bot):
    """Обработчик кнопки 'Завершить диалог'"""
    user = message.from_user
    user_id = str(user.id)
    log_user_action(user.id, user.username, "ended the dialog")
    dialog_manager.clear_history(user_id)
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer("✅ Диалог завершен. Буду очень рад поставленной оценке!", reply_markup=kb.main)
    await state.clear()

@router.message(F.text == 'Начать диалог')
async def start_entertainment(message: Message, state: FSMContext, bot: Bot):
    """Обработчик кнопки 'Начать диалог'"""
    user = message.from_user
    user_id = str(user.id)
    log_user_action(user.id, user.username, "started new dialog")
    dialog_manager.clear_history(user_id)
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer(
        "🔄 Вы перешли в режим диалога с ботом.\n Напишите ему свою просьбу и он вам обязательно ответит!"
    )
    await state.set_state(TripPlanning.user_request)

@router.message(F.text == 'Поставить оценку')
async def start_feedback(message: Message, state: FSMContext, bot: Bot):
    """Обработчик кнопки 'Поставить оценку'"""
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await message.answer(
        "Пожалуйста, оцените бота от 1 до 5 звезд⭐",
        reply_markup=kb.feedback_options
    )
    await state.set_state(TripPlanning.waiting_for_rating)

@router.message(TripPlanning.waiting_for_rating)
async def process_rating(message: Message, state: FSMContext, bot: Bot):
    """Обработчик оценки от пользователя"""
    user = message.from_user
    if message.text.startswith(("1", "2", "3", "4", "5")):
        rating = int(message.text[0])
        await state.update_data(rating=rating)
    
    log_user_action(user.id, user.username, f"rate - {message.text}")

    await message.answer(
        "Спасибо за вашу оценку!💫",
        reply_markup=kb.main
    )
    await state.clear()

@router.message(TripPlanning.user_request)
async def process_user_request(message: Message, state: FSMContext, bot: Bot):
    """Основной обработчик запросов пользователя"""
    user = message.from_user
    log_user_action(user.id, user.username, f"sent message: {message.text}")
    await state.set_state(TripPlanning.waiting_for_response)
    user_id = str(user.id)
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    processing_msg = await message.answer("⏳ Думаю...")
    
    try:
        response = dialog_manager.generate_response(
            username=user_id,
            user_message=message.text,
            prompt=prompt,
            prompt_paths=prompt_paths
        )
        
        await processing_msg.delete()
        log_user_action(user.id, user.username, f"received response: {response}")
        await message.answer(
            markdown_to_html(response),
            reply_markup=kb.dialog_options,
            parse_mode="HTML"
        )
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        log_user_action(user.id, user.username, error_msg)
        await processing_msg.delete()
        await message.answer(f"⚠️ Ошибка: {str(e)}")
    await state.set_state(TripPlanning.user_request)