import base64
import requests
import json
from dashscope import MultiModalConversation

def build_structured_prompt(frame_info, is_last=False):
    base = (
        f"当前是第 {frame_info['segment_index']} 段，这段的语音文本是：“{frame_info['text']}”，"
        f"语音起始时间是 {frame_info['start']} 秒，语音结束时间是 {frame_info['end']} 秒，"
        f"图像在该视频的第 {frame_info['timestamp']} 秒取得，"
    )
    if is_last:
        return base + "请你理解该图片和文本，当前是最后一个关键帧，请结合前面的关键帧和信息来总结该视频内容。"
    else:
        return base + "请你理解该图片和文本，先不描述，等待后续指令。"


def generate_video_summary(image_path, text, api_key):
    with open(image_path, 'rb') as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')

    messages = [
        {
            "role": "user",
            "content": [
                {"image": base64_image},
                {"text": text}
            ],
        }
    ]

    # 发起请求
    response = MultiModalConversation.call(
        api_key=api_key,
        model='llama-4-maverick-17b-128e-instruct',
        messages=messages
    )

    if response and response.get('output') and response['output'].get('choices'):
        return response['output']['choices'][0]['message']['content'][0]['text']
    else:
        print("❌ 错误：模型未返回有效结果")
        print("返回内容：", response)
        return "无法生成总结"
