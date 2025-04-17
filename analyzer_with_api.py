


import os
from dashscope import MultiModalConversation

messages = [
    {
        "role": "user",
        "content": [
            {
            "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
            },
            {"text": "这是什么😊?"},
        ],
    }
]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-e6f5a000ba014f92b4857a6dcd782591",
    model='llama-4-maverick-17b-128e-instruct', # 此处以llama-4-maverick-17b-128e-instruct为例，可按需更换模型名称。模型列表：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
    messages=messages)
print(f"模型第一轮输出：{response.output.choices[0].message.content[0]['text']}")
messages.append(response['output']['choices'][0]['message'])
user_msg = {"role": "user", "content": [{"text": "刚才你回答了什么"}]}
messages.append(user_msg)
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-e6f5a000ba014f92b4857a6dcd782591",
    model='llama-4-maverick-17b-128e-instruct',
    messages=messages)
print(f"模型第二轮输出：{response.output.choices[0].message.content[0]['text']}")