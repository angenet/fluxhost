import gradio as gr
import requests
from PIL import Image
import io
from transformers import pipeline
import os
import logging

# 初始化翻译模型
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

# 从环境变量中获取API URL和授权密钥，并进行验证
API_URL = os.getenv("FLUX_API_URL", "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev")
AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN", "修改为你自己的HUGGINGFACE_AUTH_TOKEN")

if not API_URL or not AUTH_TOKEN:
    logging.error("API URL and AUTH_TOKEN must be set as environment variables.")
    raise EnvironmentError("Missing required environment variables.")

headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

def translate_to_english(text):
    """
    将中文文本翻译成英文。
    :param text: 中文文本
    :return: 英文翻译
    """
    try:
        result = translator(text)[0]
        return result['translation_text']
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return None

def query(text):
    """
    发送请求到FLUX API以获取基于给定文本生成的图像。
    :param text: 用户输入的文本提示
    :return: 图像字节流
    """
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        response.raise_for_status()  # 检查响应状态码是否正常
        return response.content
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

def generate_image(prompt):
    """
    生成图像并返回PIL Image对象。
    :param prompt: 文本提示（中文）
    :return: PIL Image对象
    """
    # 翻译提示为英文
    translated_prompt = translate_to_english(prompt)
    if translated_prompt is None:
        return None
    
    # 根据翻译后的提示生成图像
    image_bytes = query(translated_prompt)
    if image_bytes is None:
        return None
    
    image = Image.open(io.BytesIO(image_bytes))
    return image

# 创建Gradio界面
iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="FLUX图像生成器",
    description="输入一段描述性的文字，将根据此描述生成相应的图像。支持中文输入。",
    article="<p style='text-align: center'>AI自动生成</p>"
)

# 启动Gradio应用
iface.launch()
