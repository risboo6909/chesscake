import pathlib
import os
import io
import sys
import chess
import joblib
import ray

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
)
from cv_board import recognize_board, from_file_object, from_path
from nn_pieces import recognize_pieces
from task_mgr import TaskManager, Task
from result import Err

parent_dir = pathlib.Path(__file__).parent.parent.resolve()
samples_dir = os.path.join(parent_dir, "samples")

ray.init()

UPLOAD_PHOTO, WHO_MOVES, RECOGNIZE = range(3)

lichess_url = 'https://lichess.org/editor/{}'.format

def load_models():
    models = []
    path_to_models = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "mlp")
    for file_name in os.listdir(path_to_models):
        if file_name.endswith(".joblib"):
            models.append(joblib.load(os.path.join(path_to_models, file_name)))

    return models


models = load_models()
task_mgr = TaskManager(models)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        text="""Send me a picture of a chessboard, please try to avoid noisy or big pictures.
        
I can better recognize sharp pictures with a chessboard in them with 2D plain pieces without or 
with the minimum number of any foreign objects.

Type /cancel to cancel the conversation.""",
    )
    return UPLOAD_PHOTO


async def upload_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    buffer = io.BytesIO()
    new_file = await update.message.photo[-1].get_file()
    await new_file.download(out=buffer)
    context.bot_data["buffer"] = buffer

    reply_keyboard = [["A1", "A8", "H1", "H8"]]

    await update.message.reply_text(
        text="What cell is located in the bottom left corner of the board?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard,
            one_time_keyboard=True,
            input_field_placeholder="A1, A8, H1 or H8?",
        ),
    )
    return WHO_MOVES


async def who_moves(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [["White", "Black"]]
    context.bot_data["bottom_left"] = update.message.text

    await update.message.reply_text(
        "Whose turn?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard,
            one_time_keyboard=True,
            input_field_placeholder="White or Black?",
        ),
    )
    return RECOGNIZE


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def recognize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    turn = update.message.text

    task = Task(
        turn,
        context.bot_data["bottom_left"],
        context.bot_data["buffer"],
        update.effective_user.id,
    )
    res = task_mgr.enqueue_task(task)
    if not res.success:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, I'm too busy right now. Try again later.",
            reply_markup=ReplyKeyboardRemove(),
        )

        return ConversationHandler.END

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text='You are {} in queue, your ticket is "{}", please wait...'.format(
            res.place, task.ticket
        ),
        reply_markup=ReplyKeyboardRemove(),
    )

    await res.event.wait()

    if isinstance(task.result[0], Err):
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=task.result[0].value
        )
    else:
        fen = task.result[0].value
        fen_url = fen.replace(" ", "_")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="FEN: {}\n\nAnalyze in lichess: {}".format(fen, lichess_url(fen_url)),
        )

    return ConversationHandler.END


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if task_mgr.job_requested(user_id):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="You've already submitted a request, please wait...",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Sorry, I didn't understand that command. Try typing /start to recognize a chessboard.",
    )


def debug_local():
    image_path = os.path.join(samples_dir, "fail3.jpg")
    cropped_squares = recognize_board(from_path(image_path))

    if isinstance(cropped_squares, Err):
        sys.exit(1)

    print("recognizing pieces...")
    result = recognize_pieces(
        models, cropped_squares.value, turn=chess.WHITE, bottom_left=chess.H8
    )


if __name__ == "__main__":

    # debug_local()

    tg_token = os.getenv('BOT_TOKEN')
    if tg_token is None:
        raise ValueError("BOT_TOKEN environment variable is not set")

    application = (
        ApplicationBuilder()
        .token(tg_token)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            UPLOAD_PHOTO: [
                MessageHandler(filters.PHOTO, upload_photo),
            ],
            WHO_MOVES: [
                MessageHandler(filters.Regex("^(A1|A8|H1|H8|a1|a8|h1|h8)$"), who_moves),
            ],
            RECOGNIZE: [
                MessageHandler(filters.Regex("^(White|Black)$"), recognize),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        block=False,
    )

    application.add_handler(conv_handler)

    unknown_handler = MessageHandler(filters.ALL, unknown)
    application.add_handler(unknown_handler)

    task_mgr.start()
    application.run_polling()
