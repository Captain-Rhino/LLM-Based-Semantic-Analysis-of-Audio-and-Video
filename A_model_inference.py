import base64
import requests
import json

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
