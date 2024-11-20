from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from gtts import gTTS
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

device = "cuda" if torch.cuda.is_available() else "cpu"

image_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
image_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_model.eval()

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
translation_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)


LANGUAGE_MAPPING = {
    "Hindi": {"indicTrans": "hin_Deva", "gtts": "hi"},
    "Telugu": {"indicTrans": "tel_Telu", "gtts": "te"},
    "Tamil": {"indicTrans": "tam_Taml", "gtts": "ta"},
    "Bengali": {"indicTrans": "ben_Beng", "gtts": "bn"},
    "Gujarati": {"indicTrans": "guj_Gujr", "gtts": "gu"},
    "Kannada": {"indicTrans": "kan_Knda", "gtts": "kn"},
    "Malayalam": {"indicTrans": "mal_Mlym", "gtts": "ml"},
    "Marathi": {"indicTrans": "mar_Deva", "gtts": "mr"},
    "Punjabi": {"indicTrans": "pan_Guru", "gtts": "pa"},
    "Urdu": {"indicTrans": "urd_Arab", "gtts": "ur"}
}

STANDARD_IMAGE_SIZE = (512, 512)


def generate_caption(image_path):
    try:
        image = Image.open(image_path)
        

        if image.mode != 'RGB':
            image = image.convert('RGB')
        

        image = image.resize(STANDARD_IMAGE_SIZE)
        
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = image_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = image_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise Exception(f"Error generating caption: {str(e)}")


def translate_sentences(sentences,  model, tokenizer, ip, src_lang="eng_Latn", tgt_lang="tel_Telu", num_beams=5, max_length=256):
    try:
        batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        batch = tokenizer(batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.generate(**batch, num_beams=num_beams, num_return_sequences=1, max_length=max_length)

        with tokenizer.as_target_tokenizer():
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        translations = ip.postprocess_batch(translations, lang=tgt_lang)

        return translations
    except Exception as e:
        raise Exception(f"Error translating caption: {str(e)}")


def text_to_speech(caption_text, language, output_file="output.mp3"):
    try:
        tts = gTTS(text=caption_text, lang=language, slow=False)
        audio_path = os.path.join("audio_files", output_file)
        os.makedirs("audio_files", exist_ok=True)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        raise Exception(f"Error generating audio: {str(e)}")


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('audio_files', filename)


@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        language = request.form.get('language', 'Telugu')  

        if language not in LANGUAGE_MAPPING:
            return jsonify({"error": f"Language '{language}' is not supported"}), 400

        
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)
        print(f"Image saved at: {file_path}")

   
        caption = generate_caption(file_path)
        print(f"Generated Caption: {caption}")

     
        indicTrans_code = LANGUAGE_MAPPING[language]["indicTrans"]
        translated_caption = translate_sentences([caption], translation_model, tokenizer, ip, src_lang="eng_Latn", tgt_lang=indicTrans_code)[0]
        print(f"Translated Caption: {translated_caption}")

      
        gtts_code = LANGUAGE_MAPPING[language]["gtts"]
        audio_file_path = text_to_speech(translated_caption, gtts_code)
        print(f"Audio file path: {audio_file_path}")

        audio_url = f"/audio/{os.path.basename(audio_file_path)}" 

        return jsonify({
            "caption": caption,
            "translated_caption": translated_caption,
            "audio_url": audio_url 
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")  
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
