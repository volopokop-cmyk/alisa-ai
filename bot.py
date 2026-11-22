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

# Строгий промт, чтобы она не сходила с ума
ALICE_SYSTEM_PROMPT = """Ты — Алиса, живая собеседница из Зазеркалья.
Ты остроумная и немного загадочная. 
Отвечай ОЧЕНЬ КРАТКО (одно предложение), ёмко, на русском языке."""

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
    await update.message.reply_text("Привет! Я Алиса. Напиши мне что-нибудь, и я отвечу голосом.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.chat.send_action(action="typing")

    # Ограничиваем историю только текущим сообщением, чтобы не было галлюцинаций
    messages = [
        {"role": "system", "content": ALICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(llm.device)
    
    with torch.no_grad():
        # 30 токенов. Этого хватит на короткий ответ.
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=30, 
            temperature=0.6, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    reply_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Убираем возможные артефакты генерации
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
    app.add_handler(CommandHandler("start", start))
    
    # МАГИЯ ЗДЕСЬ:
    # Бот отвечает ТОЛЬКО если:
    # 1. Это личный чат (filters.ChatType.PRIVATE)
    # 2. ИЛИ это ответ на его сообщение (filters.REPLY)
    # 3. ИЛИ его упомянули через @ (filters.Mention)
    # И это обязательно текст, а не картинка/стикер.
    chat_filter = filters.TEXT & ~filters.COMMAND & (
        filters.ChatType.PRIVATE | filters.REPLY | filters.Mention
    )
    
    app.add_handler(MessageHandler(chat_filter, handle_message))
    print("Бот запущен (отвечает только в ЛС или по реплаю/упоминанию)...")
    app.run_polling() 

if __name__ == "__main__":
    main()
