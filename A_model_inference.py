import base64
import requests
import time
import json
import os
import torch
from dashscope import MultiModalConversation


def summarize_video_from_all_frames(keyframes_combined, api_key, adaptor_path=None, output_summary_path=None):
    # 设置GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # 加载适配层
    adaptor = None
    if adaptor_path and os.path.exists(adaptor_path):
        from A_clip_finetune import ClipAdaptor
        adaptor = ClipAdaptor()
        adaptor.load_state_dict(torch.load(adaptor_path, map_location=device))
        adaptor.to(device)
        adaptor.eval()
        print(f"🔧 适配层已加载：{adaptor_path}")
    else:
        print("⚠️ 没有使用ClipAdaptor，直接用原始图像特征")

    #系统提示
    initial_prompt = (
        "你是一个AI音视频理解助手，专门处理视频中的音频和图像信息。"
        "你的任务是根据视频中的语音文本和关键帧图像，理解并总结整个视频的内容。"
        "你会收到视频的语音转录结果和图像帧信息，请帮助生成视频总结、提取关键事件，并回答与视频内容相关的问题。"
    )
    #初始化
    messages = [
        {
            "role":"system",    #设置角色为系统消息
            "content":[{"text":initial_prompt}]
        }
    ]

    for frame in keyframes_combined:
        if frame["mode"] == "text_guided" and "feat_path" in frame and adaptor:
            feat_data = torch.load(frame["feat_path"])
            image_feat = feat_data["image_feat"].to(device)
            if image_feat.dtype != torch.float32:
                image_feat = image_feat.float()
            image_feat = adaptor(image_feat).cpu()


        # 构造 prompt
        if frame.get("mode") == "text_guided":
            prompt = (
                f"当前是第 {frame['segment_idx']} 段，这段的语音文本是：“{frame['text']}”，"
                f"语音起始时间是 {frame['start']} 秒，语音结束时间是 {frame['end']} 秒，"
                f"图像在该视频的第 {frame['timestamp']} 秒取得，请你理解该图片和文本，保持沉默，等待后续指令。"
            )
        elif frame.get("mode") == "visual_compensate":
            prompt = (
                f"当前静默区间的视觉补偿帧，图像在该视频的第 {frame['timestamp']} 秒取得，请你理解该图片，保持沉默，等待后续指令。"
            )
        else:
            prompt = (
                f"当前是基于视觉显著性抽取的关键帧，图像在该视频的第 {frame['timestamp']} 秒取得，"
                f"请你观察该图像，理解其内容，保持沉默，等待后续指令。"
            )

        user_msg = {
            "role": "user",
            "content": [
                {"image": frame["image_path"]},
                {"text": prompt}
            ]
        }

        # 打印 prompt
        print("\n📝 Prompt:")
        print(prompt)

        # 模型记忆这一帧
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-plus-latest',
            messages=[user_msg]
        )

        if response and response.get('output') and response['output'].get('choices'):
            reply = response['output']['choices'][0]['message']['content'][0]['text']
        else:
            reply = "❌ 模型未返回有效结果"
            print("返回内容：", response)

        # 打印模型回复
        print("✅模型返回结果：")
        print(reply)

        messages.append(user_msg)
        messages.append({
            "role": "assistant",
            "content": [{"text": reply}]
        })

        time.sleep(1)  # 控制节奏

    # 最后一轮总结
    final_prompt = "请你根据以上所有图文内容，对整个视频进行总结。"
    messages.append({
        "role": "user",
        "content": [{"text": final_prompt}]
    })

    response = MultiModalConversation.call(
        api_key=api_key,
        model='qwen-vl-plus-latest',
        messages=messages
    )

    try:
        summary = response['output']['choices'][0]['message']['content'][0]['text']
        print("\n📽️ 视频总结完成：\n", summary)

        if output_summary_path:
            with open(output_summary_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
            print(f"✅ 总结已保存至：{output_summary_path}")

        return summary

    except Exception as e:
        print("❌ 模型调用失败或未返回有效结果")
        print("返回内容：", response)
        return None
