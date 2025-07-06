from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

# Основная клавиатура
main = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Начать диалог')],
                                     [KeyboardButton(text='Поставить оценку')],
                                     [KeyboardButton(text='Возможности'),
                                      KeyboardButton(text='Контакты')]],
                           resize_keyboard=True,
                           input_field_placeholder='Выбери один из вариантов')

# Клавиатура во время диалога
dialog_options = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Начать диалог снова")],
        [KeyboardButton(text="Завершить диалог")]
    ],
    resize_keyboard=True
)

# Клавиатура во время выыставления оценки
feedback_options = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="1 ⭐"), KeyboardButton(text="2 ⭐⭐")],
        [KeyboardButton(text="3 ⭐⭐⭐"), KeyboardButton(text="4 ⭐⭐⭐⭐")],
        [KeyboardButton(text="5 ⭐⭐⭐⭐⭐")],
    ],
    resize_keyboard=True
)