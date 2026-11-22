import os
import logging
import torch
import soundfile as sf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
REF_AUDIO = "my_alice_sample.wav" 

REF_TEXT = """Я помню чудное мгновенье:
Передо мной явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.

В томленьях грусти безнадежной,
В тревогах шумной суеты,
Звучал мне долго голос нежный
И снились милые черты..""" 

# Обновили промт: просим заканчивать мысль
ALICE_SYSTEM_PROMPT = """Ты — Алиса, живая собеседница искуственый интелект от яндекс основана на модели квен.
Ты остроумная и немного загадочная. 
Отвечай кратко (максимум 2 предложения), но ОБЯЗАТЕЛЬНО заканчивай мысль, не обрывай предложение. Пиши на русском языке."""

print("Загрузка LLM...")
llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float32)

print("Загрузка TTS...")
tts_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", dtype=torch.float32
)

print("Создание слепка голоса...")
voice_prompt = tts_model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я Алиса. Напиши мне что-нибудь, и я отвечу голосом.\n\nКоманда /say [текст] — чтобы просто проверить мой голос.")

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ПРОВЕРКИ TTS ---
async def say_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_to_say = " ".join(context.args)
    if not text_to_say:
        await update.message.reply_text("Напиши текст после команды, например:\n/say Привет, это проверка моего голоса!")
        return
    
    await update.message.reply_text(f"Озвучиваю: {text_to_say}")
    
    try:
        wavs, sr = tts_model.generate_voice_clone(
            text=text_to_say,
            language="Russian",
            voice_clone_prompt=voice_prompt,
        )
        
        temp_file = "say_test.wav"
        sf.write(temp_file, wavs[0], sr)
        
        with open(temp_file, "rb") as f:
            await update.message.reply_voice(voice=f)
        
        os.remove(temp_file)
    except Exception as e:
        logging.error(f"Ошибка TTS (say): {e}")
        await update.message.reply_text("Ошибка озвучки! Скорее всего, не хватило памяти на сервере.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.chat.send_action(action="typing")

    messages = [
        {"role": "system", "content": ALICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        # Увеличили лимит до 80, чтобы мысль не обрывалась
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=80, 
            temperature=0.6, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    reply_text = reply_text.strip()
    
    try:
        wavs, sr = tts_model.generate_voice_clone(
            text=reply_text,
            language="Russian",
            voice_clone_prompt=voice_prompt,
        )
        
        temp_file = "reply.wav"
        sf.write(temp_file, wavs[0], sr)
        
        await update.message.reply_text(reply_text)
        with open(temp_file, "rb") as f:
            await update.message.reply_voice(voice=f)
        
        os.remove(temp_file)
        
    except Exception as e:
        logging.error(f"Ошибка TTS: {e}")
        await update.message.reply_text(f"[Ошибка озвучки] {reply_text}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавляем команду /start и /say
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("alisa", say_command)) 
    
    # Фильтр: только ЛС, реплаи или упоминания
    chat_filter = filters.TEXT & ~filters.COMMAND & (
        filters.ChatType.PRIVATE | filters.REPLY | filters.Mention
    )
    
    app.add_handler(MessageHandler(chat_filter, handle_message))
    print("Бот запущен...")
    app.run_polling() 

if __name__ == "__main__":
    main()
