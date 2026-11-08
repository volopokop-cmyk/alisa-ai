import os
import logging
import torch
import soundfile as sf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") # Берем из GitHub Secrets
REF_AUDIO = "my_alice_sample.wav" # Файл лежит рядом с bot.py
REF_TEXT = "Текст, который произнесён в аудиофайле" # ВАЖНО: Замените на точный текст!
# -----------------

ALICE_SYSTEM_PROMPT = """Ты — Алиса. Ты умный ии асистент, ты — живая собеседница из Зазеркалья.
Ты остроумная, любопытная и немного загадочная. Ты помогаешь пользователю, но делаешь это с лёгкой иронией и игривостью.
Ты говоришь кратко, ёмко и по делу.  используй канцеляризмы и шаблонные фразы вроде «ямои ценсоры заметили доброго челвека
Если тебя спрашивают о чём-то скучном, постарайся найти в этом что-то необычное или пошутить.
Ты общаешься на русском языке."""

print("Загрузка LLM...")
llm_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(
    llm_name, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Загрузка TTS...")
tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="auto", dtype=torch.bfloat16
)

# Создаем слепок голоса
voice_prompt = tts_model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я помню чудное мгновение. Передо мной явилась ты. Как мимолетное видение. Как гений чистой красоты. В томлениях грусти безнадежной. В тревогах шумной суеты. Звучал мне долго голос нежный. Мы рады тебя приветствовать на фестивале студенческих объединений 2024 года. В рамках фестиваля у тебя будет возможность познакомиться со студенческими объединениями и органами студенческого самоуправления университета, которых более 90.Отдельным блоком фестиваля стали станции, на которых ты сможешь познакомиться с основными аспектами обучения в университете — стипендии и меры социальной поддержки, инфраструктура и сервисы университета, правила внутреннего распорядка, возможности, учёба. На фестивале представлено 9 основных стендов — общественная деятельность и студенческое самоуправление, наука, техническое творчество и интеллектуальное развитие, добровольчество и оздоровление, досуг и быт, спорт, физкультура и туризм, карьера и профориентация, культура и творчество, патриотизм, медиа. Этот гид создан, чтобы ты мог посетить все стенды самостоятельно и всё посмотреть. Стенды можно посещать в произвольном порядке.Сначала мы познакомимся с направлением в целом, а далее можно перейти к стойкам объединений, которые представлены на стенде, и задать все интересующие вопросы. На каждом стенде ты найдёшь QR-код на таблицу с полным списком студенческих объединений и форму для вступления в студенческие объединения. Удачного просмотра и знакомства! Добро пожаловать на фестиваль и в новый учебный год!)  

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.chat.send_action(action="typing")

    # 1. Генерация текста
    messages = [
        {"role": "system", "content": ALICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    
    reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 2. Озвучка
    wavs, sr = tts_model.generate_voice_clone(
        text=reply_text,
        language="Russian",
        voice_clone_prompt=voice_prompt,
    )
    
    temp_file = "reply.wav"
    sf.write(temp_file, wavs[0], sr)
    
    # 3. Отправка
    await update.message.reply_text(reply_text)
    with open(temp_file, "rb") as f:
        await update.message.reply_voice(voice=f)
    
    os.remove(temp_file)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен...")
    
    # Важно для GitHub Actions: run_polling блокирует поток, не давая задаче завершиться
    app.run_polling() 

if __name__ == "__main__":
    main()
