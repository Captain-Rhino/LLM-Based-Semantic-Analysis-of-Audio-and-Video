import streamlit as st
import os
import re
from A_audio_extractor import extract_audio_from_video
from A_audio_recognition import transcribe_audio
from A_video_analyzer import process_video
from A_model_inference import summarize_video_from_all_frames
import json
import subprocess
import time
import tempfile

# 文件保存基础路径
save_dir = "G:/videochat/my_design/streamlit_save"
os.makedirs(save_dir, exist_ok=True)

st.set_page_config(page_title="基于AI大模型的视频语义分析系统", layout="wide")

# ========== 左侧操作区 ==========
with st.sidebar:
    st.header("🛠 操作设置")

    # 上传视频
    uploaded_video = st.file_uploader("🎥 上传视频", type=["mp4"])


    #启动sensevoice服务器
    if st.button("🚀 启动本地 SenseVoice服务器"):
        server_path = "G:/videochat/my_design/start_sensevoice_server.py"
        try:
            subprocess.Popen(["python", server_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            st.success("✅ 正在启动 SenseVoice 服务器")
        except Exception as e:
            st.error(f"❌ 启动失败：{e}")

    # 大模型 API 选择
    api_choice = st.selectbox("🤖 选择大模型 API", ["DashScope", "OpenAI", "本地Qwen"])

    # 参数控制
    frame_interval = st.slider("📏 抽帧间隔（秒）", min_value=1, max_value=10, value=2)
    text_threshold = st.number_input("📚 文本抽帧字数阈值", min_value=10, value=80)


#功能触发
transcribe_triggered = False#b1用

# ========== 页面右侧功能按钮与输出展示 ==========
st.title("🎬 音视频语义分析系统")
st.markdown("上传视频并点击下方按钮执行各类分析功能。")
output_text = ""
b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("📝 文本转录"):
        transcribe_triggered = True
        if uploaded_video:
            video_raw_name = uploaded_video.name
            video_name = os.path.splitext(video_raw_name)[0]
            # --- 定义永久文件保存目录 ---
            video_dir = os.path.join(save_dir, video_name) # 使用 save_dir 变量
            os.makedirs(video_dir, exist_ok=True)

            # --- 定义需要永久保存的文件路径 ---
            audio_path = os.path.join(video_dir, f"{video_name}.mp3")
            json_path = os.path.join(video_dir, f"{video_name}_transcription.json")
            text_path = os.path.join(video_dir, f"{video_name}_clean.txt")

            temp_video_path = None # 初始化临时视频路径变量
            try:
                # --- 1. 将上传的视频写入临时文件 ---
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
                    temp_video_path = temp_video_file.name
                    # 将上传文件缓冲区的内容写入临时文件
                    temp_video_file.write(uploaded_video.getbuffer())
                    # (可选) 打印临时文件路径，方便调试
                    # st.write(f"⏳ 视频已暂存至: `{temp_video_path}` (处理后将删除)")

                # --- 2. 使用临时视频文件提取音频 (保存到永久路径 audio_path) ---
                extract_audio_from_video(temp_video_path, audio_path)

                # --- 3. 执行转录 (使用永久保存的音频文件) ---

                api_key = "sk-xxx" # 记得替换成你的实际 API Key 或从配置中读取
                transcription = transcribe_audio(audio_path, api_key)

                # --- 4. 保存原始 JSON 转录结果 (保存到永久路径 json_path) ---
                if transcription is not None:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(transcription, f, ensure_ascii=False, indent=2)

                    # --- 5. 清洗合并文本 (保存到永久路径 text_path) ---
                    clean_text = ""
                    # 检查 transcription 是否是预期的列表格式
                    if isinstance(transcription, list):
                        for seg in transcription:
                            line = seg.get("text", "")
                            # 保留一些基本标点符号，让文本更易读
                            line = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？《》“”：；（）()、…,.!?]", "", line)
                            clean_text += line + " "
                    elif isinstance(transcription, dict) and "res" in transcription:
                         # 兼容 A_audio_recognition.py 返回原始 dict 的情况
                         if isinstance(transcription["res"], list):
                             for seg in transcription["res"]:
                                 line = seg.get("text", "")
                                 line = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？《》“”：；（）()、…,.!?]", "", line)
                                 clean_text += line + " "
                         else:
                             st.error("❌ 转录结果格式不符合预期 (res is not list)。")
                             clean_text = "转录结果格式错误"
                    else:
                        st.error("❌ 转录结果格式不符合预期 (不是列表或包含'res'列表的字典)。")
                        clean_text = "转录结果格式错误" # 提供错误信息

                    clean_text = clean_text.strip()
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)

                    # 将内容写入统一输出框
                    output_text = clean_text
                else:
                    st.error("❌ 语音转录失败，无法生成后续文件。")
                    output_text = "语音转录失败" # 更新输出框状态

            except Exception as e:
                st.error(f"❌ 处理过程中发生错误: {e}")
                output_text = f"处理失败: {e}" # 在输出框显示错误
            finally:
                # --- 6. 清理临时视频文件 ---
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                        # (可选) 打印删除确认信息
                        # st.write(f"🗑️ 临时视频文件 `{temp_video_path}` 已删除。")
                    except Exception as e_del:
                        st.warning(f"⚠️ 删除临时文件 `{temp_video_path}` 失败: {e_del}")

        else:
            st.warning("⚠️ 请先上传视频文件")
with b2:
    if st.button("🎭 情绪识别"):
        if uploaded_video:
            video_raw_name = uploaded_video.name
            video_name = os.path.splitext(video_raw_name)[0]
            video_dir = os.path.join("G:/videochat/my_design/streamlit_save", video_name)
            os.makedirs(video_dir, exist_ok=True)

            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            audio_path = os.path.join(video_dir, f"{video_name}.mp3")
            json_path = os.path.join(video_dir, f"{video_name}_transcription.json")

            # 如果 JSON 不存在，自动转录一次
            if not os.path.exists(json_path):
                # 保存视频
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                # 提取音频
                extract_audio_from_video(video_path, audio_path)
                # 语音识别
                api_key = "sk-xxx"
                transcription = transcribe_audio(audio_path, api_key)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(transcription, f, ensure_ascii=False, indent=2)
            else:
                with open(json_path, "r", encoding="utf-8") as f:
                    transcription = json.load(f)

            # 提取情绪并格式化输出
            emotion_lines = []
            for seg in transcription:
                start = seg.get("start")
                end = seg.get("end")
                text = seg.get("text", "")
                emotions = re.findall(r"[😀-🙏✨💥❤️🌟😊😂😢😭😡😠👍👎💔🤔😳😱🤯😴😐😬😇😅😆🥺🥰😍😘😞😓😩😤😨😰😈😶]", text)
                for emo in emotions:
                    line = f"在 {start:.2f} 到 {end:.2f} 秒内，检测到 {emo} 情绪"
                    emotion_lines.append(line)

            if emotion_lines:
                output_text = "检测结果：\n" + "\n".join(emotion_lines)
            else:
                output_text = "未检测到明显情绪符号。"

        else:
            st.warning("⚠️ 请先上传视频文件")



with b3:
    if st.button("📽️ 视频总结"):
        if uploaded_video:
            video_name = os.path.splitext(uploaded_video.name)[0]
            video_dir = os.path.join("G:/videochat/my_design/streamlit_save", video_name)
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            json_path = os.path.join(video_dir, f"{video_name}_transcription.json")
            keyframe_json_path = os.path.join(video_dir, "keyframes_combined.json")
            summary_path = os.path.join(video_dir, f"{video_name}_summary.json")

            # 若 keyframes 不存在，则自动调用全流程处理生成
            if not os.path.exists(keyframe_json_path):
                st.write("⚙️ 正在抽取关键帧并生成 keyframes_combined.json ...")

                # 保存视频（如未保存）
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # 调用自动处理
                process_video(
                    video_path=video_path,
                    output_dir=video_dir,
                    api_key="sk-e6f5a000ba014f92b4857a6dcd782591",
                    do_finetune=False
                )

            # 加载关键帧图文结构
            if os.path.exists(keyframe_json_path):
                with open(keyframe_json_path, "r", encoding="utf-8") as f:
                    keyframes_combined = json.load(f)

                summary = summarize_video_from_all_frames(
                    keyframes_combined=keyframes_combined,
                    api_key="sk-e6f5a000ba014f92b4857a6dcd782591",
                    output_summary_path=summary_path
                )

                if summary:
                    output_text = "📽️ 视频总结：\n" + summary
                else:
                    output_text = "❌ 大模型未返回有效总结内容，请检查关键帧是否异常。"
            else:
                output_text = "❌ 无法获取关键帧信息，视频总结失败。"
        else:
            st.warning("⚠️ 请先上传视频文件")

with b4:
    if st.button("☁️ 生成词云图"):
        st.success("✅ 词云图已生成（演示）")


# 路径信息输出 (修改为只显示实际保存的文件)
if transcribe_triggered and uploaded_video:
    path_display = st.empty()
    with path_display.container():
        # 不再显示 video_path
        # st.write(f"🎞️ 视频已保存：`{video_path}`")
        if 'audio_path' in locals() and os.path.exists(audio_path): # 检查文件是否真的生成了
            st.write(f"🎧 音频已保存：`{audio_path}`")
        if 'json_path' in locals() and os.path.exists(json_path):
            st.write(f"📝 原始转录 JSON：`{json_path}`")
        if 'text_path' in locals() and os.path.exists(text_path):
            st.write(f"📄 清洗后文本：`{text_path}`")
    # 可以考虑让路径信息停留更久或不消失
    # time.sleep(5)
    # path_display.empty()

# 总结输出 + 对话区 (保持不变)
#st.markdown("### 🤖 视频总结与智能问答")
output_text_area = st.text_area("📤 输出框", value=output_text, height=200) # 使用新变量名避免冲突
question = st.text_input("💬 你想问这个视频什么？")
if st.button("💡 提交问题"):
    answer = f"这是对你提问“{question}”的模拟回答（暂未接 API）"
    st.text_area("🧠 大模型回答", value=answer, height=100)