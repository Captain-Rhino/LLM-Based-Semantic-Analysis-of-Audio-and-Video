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
            video_dir = os.path.join("G:/videochat/my_design/streamlit_save", video_name)
            os.makedirs(video_dir, exist_ok=True)

            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            audio_path = os.path.join(video_dir, f"{video_name}.mp3")
            json_path = os.path.join(video_dir, f"{video_name}_transcription.json")
            text_path = os.path.join(video_dir, f"{video_name}_clean.txt")

            # 保存视频
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            # 音频提取
            extract_audio_from_video(video_path, audio_path)

            # 执行转录
            api_key = "sk-xxx"
            transcription = transcribe_audio(audio_path, api_key)

            # 保存原始 JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)

            # 清洗合并文本
            clean_text = ""
            for seg in transcription:
                line = seg.get("text", "")
                line = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？《》“”：；（）()、…]", "", line)
                clean_text += line + " "

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(clean_text.strip())

            # ✅ 将内容写入统一输出框
            output_text = clean_text.strip()

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
                output_text = "🎭 检测结果：\n" + "\n".join(emotion_lines)
            else:
                output_text = "🎭 未检测到明显情绪符号。"

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


#路径信息输出
if transcribe_triggered:
    path_display = st.empty()
    with path_display.container():
        st.write(f"🎞️ 视频已保存：`{video_path}`")
        st.write(f"🎧 音频已提取：`{audio_path}`")
        st.write(f"📝 原始转录 JSON：`{json_path}`")
    time.sleep(3)
    path_display.empty()
# 总结输出 + 对话区
st.markdown("### 🤖 视频总结与智能问答")

# ✅ 统一输出框（文本转录 / 总结 / 情绪结果）
output_text = st.text_area("📤 输出框", value=output_text, height=200)

# 对话交互区
question = st.text_input("💬 你想问这个视频什么？")

if st.button("💡 提交问题"):
    answer = f"这是对你提问“{question}”的模拟回答（暂未接 API）"
    st.text_area("🧠 大模型回答", value=answer, height=100)