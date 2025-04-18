import base64
import requests
import time
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
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": text}
            ],
        }
    ]

    # 发起请求
    response = MultiModalConversation.call(
        api_key=api_key,
        model='qwen-vl-plus-latest',
        messages=messages
    )

    print("🎬 视频内容总结：", json.dumps(response, ensure_ascii=False, indent=2))

    if response and response.get('output') and response['output'].get('choices'):
        return response['output']['choices'][0]['message']['content'][0]['text']
    else:
        print("❌ 错误：模型未返回有效结果")
        print("返回内容：", response)
        return "无法生成总结"

def summarize_video_from_all_frames(keyframes_combined, api_key, output_summary_path=None):
    messages = []
    for i, frame in enumerate(keyframes_combined):
        prompt = (
            f"当前是第 {frame['segment_index']} 段，这段的语音文本是：“{frame['text']}”，"
            f"语音起始时间是 {frame['start']} 秒，语音结束时间是 {frame['end']} 秒，"
            f"图像在该视频的第 {frame['timestamp']} 秒取得，请你理解该图片和文本，先不描述，等待后续指令。"
        )

        print(prompt)

        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-plus-latest',
            messages=[{
                "role": "user",
                "content": [
                    {"image": frame['image_path']},
                    {"text": prompt}
                ]
            }]
        )

        if response and response.get('output') and response['output'].get('choices'):
            reply = response['output']['choices'][0]['message']['content'][0]['text']
        else:
            reply = "❌ 错误：模型未返回有效结果"
        print("🎬 视频内容总结：", reply)
        #time.sleep(1)

        messages.append({
            "role": "user",
            "content": [
                {"image": frame['image_path']},
                {"text": prompt}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": [{"text": reply}]
        })

    messages.append({
        "role": "user",
        "content": [{"text": "请你根据以上所有图文内容，对整个视频进行总结。"}]
    })

    response = MultiModalConversation.call(
        api_key=api_key,
        model='qwen-vl-plus-latest',
        messages=messages
    )

    #print("\n🧠 最后一轮总结调用响应：", json.dumps(response, ensure_ascii=False, indent=2))

    try:
        summary = response['output']['choices'][0]['message']['content'][0]['text']
        print("\n📽️ 视频总结完成：\n", summary)

        if output_summary_path:
            with open(output_summary_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 总结已保存至：{output_summary_path}")

        return summary

    except Exception as e:
        print("❌ 模型调用失败或未返回有效结果")
        print("返回内容：", response)
        return None
