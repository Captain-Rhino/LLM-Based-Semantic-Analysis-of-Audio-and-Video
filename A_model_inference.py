# --- START OF FILE A_model_inference.py ---

import base64
import requests
import time
import json
import os
import torch
from dashscope import MultiModalConversation # 导入 DashScope 多模态会话库
# 尝试导入 CLIP Adaptor 类
try:
    from A_clip_finetune import ClipAdaptor
except ImportError:
    # 定义一个虚拟类，以防 A_clip_finetune.py 不存在或 ClipAdaptor 未定义
    class ClipAdaptor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1) # 包含一个无意义的层
        def forward(self, x):
            print("警告: 正在使用虚拟 ClipAdaptor。")
            return x

# --- 函数：处理所有关键帧并构建对话上下文 ---
def build_video_context(keyframes_combined, api_key, adaptor_path=None):
    """
    处理视频的所有关键帧（图像和关联文本），与大模型逐帧交互，
    构建包含视频内容的对话历史（上下文）。

    Args:
        keyframes_combined (list): 包含关键帧信息的列表，每个元素是一个字典。
                                   字典应包含 'image_path', 'mode', 'text'(可选), 'timestamp' 等键。
        api_key (str): 用于调用大模型 API 的密钥。
        adaptor_path (str, optional): CLIP Adaptor 模型的权重文件路径。如果提供且有效，
                                      会尝试加载并用于处理图像特征（如果特征路径存在）。默认为 None。

    Returns:
        list: 包含系统提示、用户输入（图文帧）和模型确认回复的消息列表。
              如果处理过程中发生严重错误，可能返回 None 或部分上下文。
    """
    # 设置计算设备 (GPU 优先)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[模型推理] 推理设备设置为: {device}")

    # --- 加载 CLIP 适配层 (Adaptor) ---
    adaptor = None # 初始化适配层为 None
    if adaptor_path and os.path.exists(adaptor_path):
        print(f"[模型推理] 尝试加载适配层: {adaptor_path}")
        adaptor = ClipAdaptor() # 创建适配层实例
        try:
            # 加载预训练权重
            adaptor.load_state_dict(torch.load(adaptor_path, map_location=device))
            adaptor.to(device) # 将模型移动到指定设备
            adaptor.eval()     # 设置为评估模式
            print(f"   ✅ 适配层已成功加载到 {device}。")
        except Exception as e:
            print(f"   ❌ 加载适配层失败: {e}. 将不使用适配层。")
            adaptor = None # 加载失败则重置为 None
    else:
        if adaptor_path:
            print(f"[模型推理] ⚠️ 未找到适配层文件 ({adaptor_path})，将不使用适配层。")
        else:
             print("[模型推理] ⚠️ 未提供适配层路径，将不使用适配层。")


    # --- 系统提示 (System Prompt) ---
    # 指导大模型扮演的角色和任务
    initial_prompt = (
        "你是一个专业的AI视频内容分析助手。你的任务是仔细观察并理解按时间顺序给出的视频关键帧图像和对应的语音转录文本（如果有）。"
        "请在接收每一帧信息时，专注于理解当前帧的核心内容，并用简短的话语确认你已接收和理解（例如，“已接收帧 X，内容是...”或“已理解”），不要进行总结或回答问题。"
        "在等待到我的总结或者问答请求指令后，我会明确要求你进行总结或回答特定问题。"
    )
    # 初始化消息列表，包含系统提示
    messages = [
        {
            "role":"system",    # 角色：系统
            "content":[{"text":initial_prompt}]
        }
    ]

    print(f"[模型推理] 开始处理 {len(keyframes_combined)} 个关键帧以构建上下文...")
    # --- 遍历处理每个关键帧 ---
    for i, frame in enumerate(keyframes_combined):
        print(f"\n--- 处理帧 {i+1}/{len(keyframes_combined)} ---")
        image_feat_adapted = None # 重置该帧的适配层处理后特征

        # --- （可选）处理和应用 Adaptor ---
        # 这个部分是基于原始代码逻辑，假设关键帧字典中可能包含 'feat_path'
        # 注意：下面的模型调用 MultiModalConversation.call 通常只接受图像路径，
        # 不直接接受特征向量。这里的特征处理可能是为了其他目的或需要适配API。
        feat_path = frame.get("feat_path")
        if feat_path and adaptor and os.path.exists(feat_path):
            # 只有当特征路径存在、适配层加载成功、且文件存在时才处理
            print(f"   检测到特征文件: {feat_path}，尝试应用适配层...")
            try:
                # 加载特征数据
                feat_data = torch.load(feat_path, map_location=device)
                if "image_feat" in feat_data:
                    image_feat = feat_data["image_feat"] # 获取图像特征
                    # 确保数据类型为 float32
                    if image_feat.dtype != torch.float32:
                        image_feat = image_feat.float()

                    # 应用适配层
                    with torch.no_grad(): # 关闭梯度计算
                         image_feat_adapted = adaptor(image_feat).cpu() # 处理后移到 CPU (如果后续需要)
                    print("      ✅ 适配层已应用于特征。")
                    # 注意：image_feat_adapted 并未在后续调用中直接使用
                else:
                    print(f"      ⚠️ 特征文件 {feat_path} 中缺少 'image_feat' 键。")
            except Exception as e:
                print(f"      ❌ 处理特征文件 {feat_path} 时出错: {e}")
        # else:
        #     if feat_path and not adaptor: print("   特征文件存在但适配层未加载，跳过适配层处理。")
        #     elif feat_path and not os.path.exists(feat_path): print(f"   特征文件路径无效: {feat_path}")


        # --- 构造发送给模型的文本提示 ---
        text_prompt_content = "" # 初始化文本内容
        frame_mode = frame.get("mode", "unknown")   # 获取帧模式
        timestamp = frame.get('timestamp', '?')     # 获取时间戳
        image_path = frame.get("image_path", "")    # 获取图像路径

        # 检查图像路径是否有效
        if not image_path or not os.path.exists(image_path):
             print(f"   ❌ 错误：帧 {i+1} 的图像文件路径 '{image_path}' 无效或文件不存在，跳过此帧。")
             continue # 跳过无法处理的帧

        # 根据帧模式生成不同的提示文本
        if frame_mode == "text_guided":
            # 文本引导帧：包含文本、时间等信息
            segment_idx = frame.get('segment_idx', 'N/A')
            text = frame.get('text', '无文本').strip()
            start_time = frame.get('start', '?')
            end_time = frame.get('end', '?')
            similarity = frame.get('similarity', 'N/A') # 获取相似度（如果存在）
            text_prompt_content = (
                f"帧 {i+1}/{len(keyframes_combined)} (文本引导模式): \n"
                f"对应语音段: {segment_idx} (时间: {start_time}s - {end_time}s)\n"
                f"语音文本: “{text}”\n"
                f"图像时间戳: {timestamp}s\n"
                # f"图文相似度: {similarity}\n" # (可选) 加入相似度信息
                f"请理解以上图文信息并确认接收。"
            )
        elif frame_mode == "visual_compensate":
            # 视觉补偿帧：通常在静默区间
            start = frame.get('start', '?')
            end = frame.get('end', '?')
            text_prompt_content = (
                f"帧 {i+1}/{len(keyframes_combined)} (视觉补偿模式): \n"
                f"位于静默区间: {start}s - {end}s\n"
                f"图像时间戳: {timestamp}s\n"
                f"请理解图像内容并确认接收。"
            )
        elif frame_mode == "visual_guided":
             # 纯视觉帧
             importance = frame.get('importance', 'N/A')
             text_prompt_content = (
                 f"帧 {i+1}/{len(keyframes_combined)} (纯视觉模式): \n"
                 f"图像时间戳: {timestamp}s\n"
                 # f"视觉重要性: {importance}\n" # (可选) 加入重要性信息
                 f"请观察图像并确认接收。"
             )
        else: # 其他未知模式
            text_prompt_content = (
                f"帧 {i+1}/{len(keyframes_combined)} (模式: {frame_mode}): \n"
                f"图像时间戳: {timestamp}s\n"
                f"请理解图像内容并确认接收。"
            )

        # --- 准备发送给大模型的消息体 ---
        user_msg_content = [
            {"image": image_path},      # 图像内容，使用文件路径
            {"text": text_prompt_content} # 文本提示
        ]
        user_msg = {
            "role": "user", # 角色：用户
            "content": user_msg_content
        }
        #测试时间1
        test_start_1 = time.time()
        # --- 打印将要发送的内容 (调试用) ---
        print(f"\n   📝 发送给模型 (帧 {i+1}):")
        print(f"      图像: {image_path}")
        print(f"      文本: {text_prompt_content}")

        # --- 调用大模型 API 处理当前帧 ---
        try:
            # 只发送当前用户消息，让模型处理这一帧的信息
            response = MultiModalConversation.call(
                api_key=api_key,              # 使用传入的 API Key
                model='qwen-vl-plus-latest', # 指定模型 (确保可用)
                messages=[user_msg]           # 只包含当前帧的消息
            )

            # --- 解析模型的回复 ---
            reply_text = "模型未返回有效确认文本。" # 设置默认回复
            # 严格检查返回结果的结构
            if (response and isinstance(response, dict) and
                response.get('status_code') == 200 and # 状态码 200 表示成功
                response.get('output') and isinstance(response['output'], dict) and
                response['output'].get('choices') and isinstance(response['output']['choices'], list) and
                len(response['output']['choices']) > 0 and # choices 列表不为空
                response['output']['choices'][0].get('message') and isinstance(response['output']['choices'][0]['message'], dict) and
                response['output']['choices'][0]['message'].get('content') and isinstance(response['output']['choices'][0]['message']['content'], list) and
                len(response['output']['choices'][0]['message']['content']) > 0 and # content 列表不为空
                response['output']['choices'][0]['message']['content'][0].get('text')): # 文本内容存在

                # 提取模型回复的文本
                reply_text = response['output']['choices'][0]['message']['content'][0]['text']
                print(f"   ✅ 模型回复 (帧 {i+1}): {reply_text[:150]}...") # 打印回复的前150个字符
                #测试时间2
                test_start_2 = time.time()
                print(f"api处理时间：{test_start_2 - test_start_1:.4f} 秒")
            else:
                # 如果响应结构不符合预期，打印警告和完整的响应内容
                print(f"   ⚠️ 模型返回结构异常或无有效文本确认 (帧 {i+1})。Response: {response}")

        except Exception as e:
            # 捕获调用 API 时的异常
            print(f"   ❌ 调用模型处理帧 {i+1} 时出错: {e}")
            # 记录错误信息作为模型的回复
            reply_text = f"处理帧时发生错误: {e}"

        # --- 将用户的消息和模型的回复（或错误信息）添加到总的对话历史中 ---
        messages.append(user_msg) # 添加用户发送的消息
        messages.append({
            "role": "assistant", # 角色：助手 (代表模型的回复)
            "content": [{"text": reply_text}]
        })

        # 控制 API 调用频率，防止过于频繁请求 (根据 API 提供商的限制调整)
        time.sleep(1)

    print("\n[模型推理] ✅ 所有帧处理完毕，视频上下文已构建完成。")
    return messages # 返回包含完整对话历史的消息列表

# --- 函数：基于构建好的上下文生成总结 ---
def generate_summary_from_context(messages_context, api_key, output_summary_path=None):
    """
    使用预先构建好的包含视频内容的对话上下文，请求大模型生成视频总结。

    Args:
        messages_context (list): 包含完整对话历史的消息列表。
        api_key (str): API 密钥。
        output_summary_path (str, optional): 如果提供，则将生成的总结保存到此路径的 JSON 文件中。

    Returns:
        str: 生成的视频总结文本。如果失败则返回 None。
    """
    if not messages_context:
        print("[模型推理] ❌ 无法生成总结，因为输入的上下文为空。")
        return None

    # 创建上下文列表的副本，以免修改原始列表
    summary_messages = list(messages_context)

    # --- 添加最终的总结指令 ---
    final_prompt = "现在，请根据以上我们交互的所有视频帧图像、对应的文本信息以及你的理解，对整个视频内容进行一个全面、连贯的总结。请直接输出总结内容，避免包含诸如“好的，这是总结：”或“总结如下：”等额外语句。"
    summary_messages.append({
        "role": "user", # 角色：用户 (提出总结请求)
        "content": [{"text": final_prompt}]
    })

    print("\n[模型推理] 💡 正在请求模型生成视频总结...")
    try:
        # 调用大模型 API，发送包含完整历史和总结请求的消息列表
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-plus-latest',
            messages=summary_messages # 发送完整上下文
        )

        # --- 解析总结响应 ---
        summary = None # 初始化总结变量
        # 同样进行严格的结构检查
        if (response and isinstance(response, dict) and
            response.get('status_code') == 200 and
            response.get('output') and isinstance(response['output'], dict) and
            response['output'].get('choices') and isinstance(response['output']['choices'], list) and
            len(response['output']['choices']) > 0 and
            response['output']['choices'][0].get('message') and isinstance(response['output']['choices'][0]['message'], dict) and
            response['output']['choices'][0]['message'].get('content') and isinstance(response['output']['choices'][0]['message']['content'], list) and
            len(response['output']['choices'][0]['message']['content']) > 0 and
            response['output']['choices'][0]['message']['content'][0].get('text')):

            # 提取总结文本
            summary = response['output']['choices'][0]['message']['content'][0]['text'].strip() # 去除可能的首尾空格
            print(f"[模型推理] ✅ 视频总结完成:\n{summary}")

            # --- 如果指定了输出路径，则保存总结 ---
            if output_summary_path:
                 print(f"   💾 正在尝试保存总结到: {output_summary_path}")
                 # 确保输出目录存在
                 summary_dir = os.path.dirname(output_summary_path)
                 if summary_dir: # 检查目录名非空
                     os.makedirs(summary_dir, exist_ok=True)

                 try:
                     # 将总结保存为 {"summary": "总结内容..."} 格式的 JSON
                     with open(output_summary_path, "w", encoding="utf-8") as f:
                         json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
                     print(f"      ✅ 总结已成功保存。")
                 except Exception as e_write:
                     print(f"      ❌ 保存总结文件到 {output_summary_path} 时出错: {e_write}")

            return summary # 返回总结文本
        else:
            # 如果响应结构不符合预期
            print(f"[模型推理] ❌ 模型未返回有效的总结内容。Response: {response}")
            return None

    except Exception as e:
        # 如果调用 API 生成总结时出错
        print(f"[模型推理] ❌ 调用模型生成总结时出错: {e}")
        return None

# --- 新增函数：基于上下文进行问答 ---
def ask_question_about_video(messages_context, question, api_key):
    """
    使用预先构建好的视频对话上下文，向大模型提出关于视频内容的具体问题。

    Args:
        messages_context (list): 包含视频内容的对话历史消息列表。
        question (str): 用户提出的关于视频内容的问题。
        api_key (str): API 密钥。

    Returns:
        str: 大模型针对问题的回答。如果出错或无法回答，则返回相应的错误信息。
    """
    # --- 输入检查 ---
    if not messages_context:
        print("[模型推理] ❌ 无法回答问题，因为视频上下文为空。")
        return "错误：视频上下文尚未建立，请先运行“视频总结”功能。"
    if not question or not question.strip(): # 检查问题是否为空或仅包含空格
        print("[模型推理] ❌ 问题为空，无法提问。")
        return "错误：请输入您想问的问题。"

    # --- 准备问答消息列表 ---
    # 创建上下文列表的副本，避免影响原始上下文
    qa_messages = list(messages_context)

    # --- 添加用户的问题 ---
    # 可以考虑在问题前加上引导，帮助模型聚焦
    qa_prompt = f"基于以上你对视频内容的理解，请回答以下问题：\n\n问题：{question}\n\n请直接给出答案。"
    qa_messages.append({
        "role": "user", # 角色：用户 (提出问题)
        "content": [{"text": qa_prompt}]
        # "content": [{"text": question}] # 或者直接发送问题
    })

    print(f"\n[模型推理] ❓ 正在向模型提问：{question}")
    try:
        # --- 调用大模型进行问答 ---
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen-vl-plus-latest', # 使用与之前相同的模型
            messages=qa_messages       # 发送包含上下文和问题的消息列表
        )

        # --- 解析问答响应 ---
        answer = f"模型未能回答该问题 (状态码: {response.get('status_code', 'N/A')})。" # 默认错误回答
        # 严格检查响应结构
        if (response and isinstance(response, dict) and
            response.get('status_code') == 200 and
            response.get('output') and isinstance(response['output'], dict) and
            response['output'].get('choices') and isinstance(response['output']['choices'], list) and
            len(response['output']['choices']) > 0 and
            response['output']['choices'][0].get('message') and isinstance(response['output']['choices'][0]['message'], dict) and
            response['output']['choices'][0]['message'].get('content') and isinstance(response['output']['choices'][0]['message']['content'], list) and
            len(response['output']['choices'][0]['message']['content']) > 0 and
            response['output']['choices'][0]['message']['content'][0].get('text')):

            # 提取答案文本
            answer = response['output']['choices'][0]['message']['content'][0]['text'].strip()
            print(f"[模型推理] 💡 模型回答:\n{answer}")

        else:
            # 如果响应结构不符合预期
            print(f"[模型推理] ❌ 模型未返回有效的回答。Response: {response}")

        return answer # 返回答案文本或错误信息

    except Exception as e:
        # 如果调用 API 回答问题时出错
        print(f"[模型推理] ❌ 调用模型回答问题时出错: {e}")
        return f"调用API回答问题时发生错误: {e}"


# --- 重构后的原始函数 (可以保留作为整体流程的调用入口) ---
def summarize_video_from_all_frames(keyframes_combined, api_key, adaptor_path=None, output_summary_path=None):
    """
    （重构后）执行完整的视频处理流程：构建上下文 -> 生成总结。

    Args:
        keyframes_combined (list): 关键帧数据列表。
        api_key (str): API 密钥。
        adaptor_path (str, optional): Adaptor 路径。
        output_summary_path (str, optional): 总结保存路径。

    Returns:
        tuple: (总结文本, 消息上下文列表) 或 (None, None) 如果失败。
               返回上下文是为了让调用者（如 Streamlit UI）可以保存它用于后续问答。
    """
    print("--- 开始执行完整流程：构建上下文并生成总结 ---")
    # 步骤 1: 构建视频上下文
    print("   步骤 1: 构建视频上下文...")
    messages_context = build_video_context(keyframes_combined, api_key, adaptor_path)

    # 步骤 2: 基于上下文生成总结 (仅当上下文构建成功时)
    if messages_context:
        print("\n   步骤 2: 基于上下文生成视频总结...")
        summary = generate_summary_from_context(messages_context, api_key, output_summary_path)
        # 返回总结文本和构建好的上下文
        return summary, messages_context
    else:
        # 如果上下文构建失败，则无法进行总结
        print("❌ 构建视频上下文失败，无法继续生成总结。")
        # 返回 None 表示失败
        return None, None