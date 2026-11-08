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

# 1. Файл с вашим голосом (короткий, 3-10 секунд, чистый WAV)
REF_AUDIO = "my_alice_sample.wav" 

# 2. ТОЧНЫЙ текст того, что вы говорите в этом файле (слово в слово!)
REF_TEXT = "Привет, меня зовут Алиса, и я рада тебя видеть." 
# -----------------

# 3. Системный промт: только характер Алисы, без лишней информации
ALICE_SYSTEM_PROMPT = """Ты — Алиса. Ты умный ИИ-ассистент и живая собеседница из Зазеркалья.
Ты остроумная, любопытная и немного загадочная. Ты помогаешь пользователю, но делаешь это с лёгкой иронией и игривостью.
Ты говоришь кратко, ёмко и по делу. Не используй канцеляризмы и шаблонные фразы.
Если тебя спрашивают о чём-то скучном, постарайся найти в этом что-то необычное или пошутить.
Ты общаешься на русском языке."""

print("Загрузка LLM (мозга Алисы)...")
llm_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
# Используем float16 для экономии памяти на GitHub Actions
llm = AutoModelForCausalLM.from_pretrained(
    llm_name, torch_dtype=torch.float16, device_map="auto"
)

print("Загрузка TTS (голоса Алисы)...")
tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="auto", dtype=torch.float16
)

print("Создание слепка голоса...")
# Создаем слепок голоса ОДИН раз при запуске
voice_prompt = tts_model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я помню чудное мгновение. Передо мной явилась ты. Как мимолетное видение. Как гений чистой красоты. В томлениях грусти безнадежной. В тревогах шумной суеты. Звучал мне долго голос нежный.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.chat.send_action(action="typing")

    # 1. Генерация текста через Qwen
    messages = [
        {"role": "system", "content": ALICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        # max_new_tokens=150 защищает от слишком длинных ответов, которые TTS не осилит
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 2. Озвучка через Qwen TTS
    try:
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
        
    except Exception as e:
        # Если TTS упал (например, не хватило памяти), отправляем хотя бы текст
        logging.error(f"Ошибка TTS: {e}")
        await update.message.reply_text(f"[Ошибка озвучки] {reply_text}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен и готов к работе...")
    
    app.run_polling() 

if __name__ == "__main__":
    main()
