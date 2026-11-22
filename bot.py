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

# Твой текст из 14-секундного аудио
REF_TEXT = """Я помню чудное мгновенье:
Передо мной явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.

В томленьях грусти безнадежной,
В тревогах шумной суеты,
Звучал мне долго голос нежный
И снились милые черты..""" 

# ПРОМТ: Строго без стиха!
ALICE_SYSTEM_PROMPT = """Ты — Алиса, живая собеседница из Зазеркалья.
Ты остроумная и немного загадочная. Отвечай кратко, ёмко, по делу, на русском языке."""

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
    await update.message.reply_text("Привет! Я Алиса. Напиши мне что-нибудь.")

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
        # 20 токенов - это 1-2 коротких предложения. TTS успеет их озвучить.
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=20, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
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
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен...")
    app.run_polling() 

if __name__ == "__main__":
    main()
