# --- START OF FILE A_streamlit_UI.py ---

import streamlit as st
import os
import re
import torch
from A_audio_extractor import extract_audio_from_video
from A_audio_recognition import transcribe_audio
from A_model_inference import build_video_context, generate_summary_from_context, ask_question_about_video
from A_keyframe_extractor import KeyframeExtractor
import json
import subprocess
import time
import tempfile
import traceback
# 尝试导入可视化函数，如果失败则禁用相关按钮
try:
    # 同时导入词云图和思维导图函数
    from A_visualizer import generate_wordcloud, generate_mindmap_from_summary
    visualizer_available = True
    # 额外检查 graphviz 是否可用，它是思维导图的核心依赖
    try:
        import graphviz
        mindmap_available = True
        print("Graphviz 可用，思维导图功能已启用。")
    except ImportError:
        mindmap_available = False
        print("警告：无法导入 graphviz 库，思维导图功能将不可用。请确保已安装 graphviz 并将其添加到系统 PATH。")
except ImportError:
    visualizer_available = False
    mindmap_available = False # 如果 A_visualizer 都导入不了，那肯定用不了
    print("警告：无法导入 A_visualizer，词云图和思维导图功能将不可用。")


# --- 全局设置 (保持不变) ---
save_dir = r"G:\videochat\my_design\streamlit_save"
os.makedirs(save_dir, exist_ok=True)
if 'api_key' not in st.session_state:
    st.session_state.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # ！！重要：替换！！

st.set_page_config(page_title="基于AI大模型的视频语义分析系统", layout="wide")

# --- 侧边栏 (保持不变) ---
with st.sidebar:
    st.header("🛠️ 操作设置")
    uploaded_video = st.file_uploader("🎥 上传本地视频文件", type=["mp4"], key="video_uploader")

    if uploaded_video and st.session_state.get('current_video_name') != uploaded_video.name:
        st.info(f"✨ 新视频: {uploaded_video.name}，重置状态。")
        st.session_state.current_video_name = uploaded_video.name
        keys_to_clear = ['video_context', 'video_summary', 'output_text',
                         'transcription_done', 'keyframes_done', 'qa_answer',
                         'audio_path', 'json_path', 'text_path', 'keyframe_json_path',
                         'summary_path', 'video_path', 'video_dir']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]

    if st.button("🚀 启动 SenseVoice"): # 简化显示文本
        server_path = r"G:\videochat\my_design\start_sensevoice_server.py"
        try:
            subprocess.Popen(["python", os.path.normpath(server_path)], creationflags=subprocess.CREATE_NEW_CONSOLE)
            st.success("✅ SenseVoice 启动指令已发送。")
        except Exception as e: st.error(f"❌ 启动失败: {e}")

    st.session_state.api_key = st.text_input("🔑 DashScope API Key", value=st.session_state.api_key, type="password", help="...")
    st.subheader("⚙️ 关键帧提取参数")
    st.session_state.frame_interval = st.slider("视觉间隔(秒)", 1, 10, st.session_state.get('frame_interval', 2), help="...")
    st.session_state.text_threshold = st.number_input("文本阈值(字/帧)", 10, 500, st.session_state.get('text_threshold', 80), help="...")

# --- 初始化 Session State (保持不变) ---
session_defaults = {
    'output_text': "", 'transcription_done': False, 'keyframes_done': False,
    'video_context': None, 'video_summary': None, 'qa_answer': "",
    'current_video_name': None, 'video_dir': None, 'video_path': None,
    'audio_path': None, 'json_path': None, 'text_path': None,
    'keyframe_json_path': None, 'clip_features_path': None, 'summary_path': None
}
for key, default_value in session_defaults.items():
    if key not in st.session_state: st.session_state[key] = default_value

# --- 主页面布局 ---
st.title("基于 AI 大模型的音视频语义分析系统") # 更新标题 Emoji
st.markdown("请在左侧上传 MP4 视频，然后按顺序点击下方按钮执行分析。")

b1, b2, b3, b4 = st.columns(4)

# === 创建状态显示占位符 ===
status_placeholder = st.empty()

# --- 文件路径动态定义 (保持不变) ---
if uploaded_video and st.session_state.current_video_name == uploaded_video.name:
    if st.session_state.video_dir is None:
        video_raw_name = uploaded_video.name
        video_name = os.path.splitext(video_raw_name)[0]
        st.session_state.video_dir = os.path.join(save_dir, video_name)
        st.session_state.video_path = os.path.join(st.session_state.video_dir, video_raw_name)
        st.session_state.audio_path = os.path.join(st.session_state.video_dir, f"{video_name}.mp3")
        st.session_state.json_path = os.path.join(st.session_state.video_dir, f"{video_name}_transcription.json")
        st.session_state.text_path = os.path.join(st.session_state.video_dir, f"{video_name}_clean.txt")
        st.session_state.keyframe_json_path = os.path.join(st.session_state.video_dir, f"{video_name}_keyframes.json")
        st.session_state.clip_features_path = os.path.join(st.session_state.video_dir, "clip_features.pth")
        st.session_state.summary_path = os.path.join(st.session_state.video_dir, f"{video_name}_summary.json")

# --- 功能按钮逻辑 (使用 st.empty) ---

# --- b1: 文本转录 (代码保持不变) ---
with b1:
    if st.button("文本转录", key="btn_transcribe"):
        if uploaded_video and st.session_state.video_dir:
            # 重置状态
            st.session_state.output_text = ""
            st.session_state.transcription_done = False
            st.session_state.keyframes_done = False
            st.session_state.video_context = None
            st.session_state.video_summary = None
            st.session_state.qa_answer = ""

            status_placeholder.info("🚀 **开始文本转录...**")
            log_messages = ["🚀 **开始文本转录...**"]
            temp_video_path = None
            try:
                os.makedirs(st.session_state.video_dir, exist_ok=True)
                log_messages.append(f"   - 准备目录: `{st.session_state.video_dir}`")
                status_placeholder.info("\n".join(log_messages))

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
                    temp_video_path = temp_video_file.name
                    temp_video_file.write(uploaded_video.getbuffer())
                log_messages.append(f"   - 视频暂存: `{temp_video_path}`")
                status_placeholder.info("\n".join(log_messages))

                log_messages.append(f"   - 提取音频: `{st.session_state.audio_path}`")
                status_placeholder.info("\n".join(log_messages))
                extract_audio_from_video(temp_video_path, st.session_state.audio_path)
                log_messages.append(f"   - ✅ 音频完成。")
                status_placeholder.info("\n".join(log_messages))

                log_messages.append(f"   - 🎙️ 语音识别...")
                status_placeholder.info("\n".join(log_messages))
                transcription_result = transcribe_audio(st.session_state.audio_path, st.session_state.api_key)
                log_messages.append(f"   - ✅ 识别调用完成。")
                status_placeholder.info("\n".join(log_messages))

                if transcription_result is not None:
                    log_messages.append(f"   - 保存 JSON: `{st.session_state.json_path}`")
                    status_placeholder.info("\n".join(log_messages))
                    with open(st.session_state.json_path, "w", encoding="utf-8") as f:
                        json.dump(transcription_result, f, ensure_ascii=False, indent=2)

                    log_messages.append(f"   - 清洗文本: `{st.session_state.text_path}`")
                    status_placeholder.info("\n".join(log_messages))
                    clean_text = ""
                    if isinstance(transcription_result, list):
                        for seg in transcription_result: clean_text += seg.get("text", "") + " "
                    elif isinstance(transcription_result, dict) and "res" in transcription_result:
                         if isinstance(transcription_result["res"], list):
                             for seg in transcription_result["res"]: clean_text += seg.get("text", "") + " "
                    clean_text = clean_text.strip()
                    clean_text = re.sub(r'字幕由(.*?)生成', '', clean_text).strip()
                    with open(st.session_state.text_path, "w", encoding="utf-8") as f: f.write(clean_text)

                    st.session_state.output_text = "转录完成。\n" + clean_text
                    st.session_state.transcription_done = True
                    log_messages.append("✅ **文本转录成功完成！**")
                    status_placeholder.success("\n".join(log_messages))
                else:
                    st.session_state.output_text = "语音转录失败。"
                    log_messages.append("❌ **语音转录失败，未收到有效结果。**")
                    status_placeholder.error("\n".join(log_messages))

            except Exception as e:
                st.session_state.output_text = f"处理失败: {e}"
                log_messages.append(f"❌ **处理文本转录时出错: {e}**")
                log_messages.append(traceback.format_exc())
                status_placeholder.error("\n".join(log_messages))
                st.error(f"❌ 处理文本转录时发生错误: {e}") # 主界面也提示
            finally:
                 if temp_video_path and os.path.exists(temp_video_path):
                     try: os.remove(temp_video_path)
                     except Exception as e_del: st.warning(f"⚠️ 删除临时文件失败: {e_del}")
        else:
            st.warning("⚠️ 请先在左侧上传视频文件。")

# --- b2: 情绪识别 (代码保持不变) ---
with b2:
    if st.button("情绪识别", key="btn_emotion", help="基于转录文本中的 Emoji 进行简单识别"):
         if not uploaded_video or not st.session_state.json_path:
             st.warning("⚠️ 请先上传视频。")
         elif not st.session_state.get('transcription_done') or not os.path.exists(st.session_state.json_path):
             st.warning("⚠️ 请先成功运行“文本转录”。")
         else:
             status_placeholder.info("👀 **开始情绪识别...**")
             log_messages = ["👀 **开始情绪识别...**"]
             try:
                 log_messages.append(f"   - 读取转录文件: `{st.session_state.json_path}`")
                 status_placeholder.info("\n".join(log_messages))
                 with open(st.session_state.json_path, "r", encoding="utf-8") as f:
                     transcription = json.load(f)

                 log_messages.append(f"   - 分析情绪符号 (Emoji)...")
                 status_placeholder.info("\n".join(log_messages))
                 emotion_lines = []
                 if isinstance(transcription, list):
                     for seg in transcription:
                         text = seg.get("text", "")
                         emotions = re.findall(r"[😀-🙏🤔-🤯😴-🧿]", text)
                         if emotions:
                             start = seg.get("start",-1); end = seg.get("end", -1)
                             line = f"时间 [{start:.2f}s - {end:.2f}s]: 检测到 {' '.join(emotions)}"
                             emotion_lines.append(line)

                 if emotion_lines:
                     result_text = "情绪符号检测结果：\n" + "\n".join(emotion_lines)
                     st.session_state.output_text = result_text
                     log_messages.append("✅ **情绪识别完成。**")
                     log_messages.append(result_text)
                     status_placeholder.success("\n".join(log_messages))
                 else:
                     st.session_state.output_text = "未检测到明显情绪符号 (Emoji)。"
                     log_messages.append("✅ **未检测到明显情绪符号。**")
                     status_placeholder.success("\n".join(log_messages))

             except Exception as e:
                 st.session_state.output_text = f"情绪识别失败: {e}"
                 log_messages.append(f"❌ **情绪识别出错: {e}**")
                 log_messages.append(traceback.format_exc())
                 status_placeholder.error("\n".join(log_messages))
                 st.error(f"❌ 情绪识别过程中发生错误: {e}")

# --- b3: 视频总结与问答准备 (代码保持不变) ---
with b3:
    if st.button("视频总结与问答", key="btn_summary"):
        required_paths = ['video_dir', 'json_path', 'keyframe_json_path', 'summary_path', 'video_path', 'audio_path']
        if not uploaded_video or not all(st.session_state.get(p) for p in required_paths):
            st.warning("⚠️ 请先上传视频文件。")
        elif not st.session_state.get('transcription_done') or not os.path.exists(st.session_state.json_path):
            st.warning("⚠️ 文本转录尚未完成，请先运行“文本转录”。")
        else:
            # 重置状态
            st.session_state.output_text = ""
            st.session_state.keyframes_done = False
            st.session_state.video_context = None
            st.session_state.video_summary = None
            st.session_state.qa_answer = ""

            status_placeholder.info("🔄 **开始视频总结与问答准备...**")
            log_messages = ["🔄 **开始视频总结与问答准备...**"]
            try:
                os.makedirs(st.session_state.video_dir, exist_ok=True)
                log_messages.append(f"   - 准备目录: `{st.session_state.video_dir}`")
                status_placeholder.info("\n".join(log_messages))

                if not os.path.exists(st.session_state.video_path):
                     log_messages.append(f"   - 保存视频文件: {st.session_state.video_path}")
                     status_placeholder.info("\n".join(log_messages))
                     with open(st.session_state.video_path, "wb") as f: f.write(uploaded_video.getbuffer())

                log_messages.append("   - 步骤 1: 加载转录数据...")
                status_placeholder.info("\n".join(log_messages))
                with open(st.session_state.json_path, "r", encoding="utf-8") as f: transcription_data = json.load(f)

                log_messages.append(f"   - 步骤 2: 提取关键帧 (视觉间隔: {st.session_state.frame_interval}s, 文本阈值: {st.session_state.text_threshold}字/帧)...")
                status_placeholder.info("\n".join(log_messages))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                extractor = KeyframeExtractor(device=device)
                keyframes_data = extractor.extract_keyframes(
                    video_path=st.session_state.video_path, output_dir=st.session_state.video_dir,
                    asr_data=transcription_data, audio_path=st.session_state.audio_path,
                    frame_interval=st.session_state.frame_interval, text_threshold=st.session_state.text_threshold
                )
                if not keyframes_data: raise ValueError("关键帧提取失败，未返回数据。")
                log_messages.append(f"     - ✅ 关键帧提取完成，共 {len(keyframes_data)} 帧。")
                status_placeholder.info("\n".join(log_messages))

                log_messages.append(f"   - 保存关键帧 JSON: {st.session_state.keyframe_json_path}")
                status_placeholder.info("\n".join(log_messages))
                with open(st.session_state.keyframe_json_path, "w", encoding="utf-8") as f: json.dump(keyframes_data, f, ensure_ascii=False, indent=2)
                st.session_state.keyframes_done = True

                log_messages.append("   - 步骤 3: 构建视频上下文 (与大模型交互)...")
                status_placeholder.info("\n".join(log_messages))
                adaptor_path_to_use = os.path.normpath(r"G:\videochat\my_design\adaptor_results\best_adaptor.pth")
                use_adaptor = False
                if os.path.exists(adaptor_path_to_use): use_adaptor = True
                log_messages.append(f"     - 使用适配层: {'是' if use_adaptor else '否'}")
                status_placeholder.info("\n".join(log_messages))
                messages_context = build_video_context(keyframes_data, st.session_state.api_key, adaptor_path_to_use if use_adaptor else None)
                if not messages_context: raise ValueError("构建视频上下文失败。")
                st.session_state.video_context = messages_context
                log_messages.append("     - ✅ 视频上下文已成功构建。")
                status_placeholder.info("\n".join(log_messages))

                log_messages.append("   - 步骤 4: 生成视频总结...")
                status_placeholder.info("\n".join(log_messages))
                summary = generate_summary_from_context(st.session_state.video_context, st.session_state.api_key, st.session_state.summary_path)
                if not summary: raise ValueError("生成视频总结失败。")
                st.session_state.video_summary = summary
                st.session_state.output_text = "视频总结:\n" + summary
                log_messages.append("     - ✅ 总结生成完成。")
                status_placeholder.info("\n".join(log_messages))

                log_messages.append("✅ **视频总结与问答准备完成！**")
                status_placeholder.success("\n".join(log_messages))

            except Exception as e:
                log_messages.append(f"❌ **处理视频总结流程时出错: {e}**")
                log_messages.append(traceback.format_exc())
                status_placeholder.error("\n".join(log_messages))
                st.error(f"❌ 处理视频总结流程时发生错误: {e}") # 主界面也提示

# --- b4: 生成思维导图 ---
with b4:
     # --- 更新：按钮和帮助文本 ---
     btn_text = "生成思维导图"
     btn_help = "基于视频总结生成思维导图 (需要 Graphviz)" if mindmap_available else "Graphviz 未安装或配置不正确，无法生成思维导图"
     btn_disabled = not mindmap_available # 如果 graphviz 不可用则禁用按钮

     if st.button(btn_text, key="btn_mindmap", help=btn_help, disabled=btn_disabled):
         # --- 前置条件检查：需要总结完成 ---
         if not uploaded_video or not st.session_state.summary_path:
             st.warning("⚠️ 请先上传视频。")
         elif not st.session_state.get('video_summary') or not os.path.exists(st.session_state.summary_path):
             st.warning("⚠️ 请先运行“视频总结与问答准备”。")
         else:
             status_placeholder.info("🧠 **开始生成思维导图...**")
             log_messages = ["🧠 **开始生成思维导图...**"]
             try:
                 # 导入已在顶部尝试过
                 log_messages.append("   - 加载总结数据...")
                 status_placeholder.info("\n".join(log_messages))
                 # generate_mindmap_from_summary 函数需要总结文件路径

                 video_name_mm = os.path.splitext(uploaded_video.name)[0]
                 log_messages.append(f"   - 调用生成函数 (视频名: {video_name_mm})...")
                 status_placeholder.info("\n".join(log_messages))

                 # --- 调用思维导图生成函数 ---
                 mindmap_output_path = generate_mindmap_from_summary(
                     summary_path=st.session_state.summary_path,
                     output_dir=st.session_state.video_dir,
                     video_name=video_name_mm
                 )
                 log_messages.append("   - 生成函数调用完成。")
                 status_placeholder.info("\n".join(log_messages))

                 # --- 检查并显示结果 ---
                 if mindmap_output_path and os.path.exists(mindmap_output_path):
                     st.session_state.output_text = f"思维导图已生成。\n保存路径: {mindmap_output_path}"
                     log_messages.append("   - ✅ 思维导图文件已生成。")
                     log_messages.append("✅ **思维导图生成成功！**")
                     status_placeholder.success("\n".join(log_messages))
                     # 图片在主界面显示
                     st.image(mindmap_output_path, caption=f"{video_name_mm} 的思维导图")
                 else:
                     st.session_state.output_text = "思维导图文件未生成。"
                     log_messages.append("❌ **思维导图文件未生成或未找到。**")
                     status_placeholder.error("\n".join(log_messages))
                     st.error("❌ 思维导图文件未生成，请检查 Graphviz 是否正确安装和配置。")

             except ImportError: # 捕获可能的 Graphviz 运行时导入错误（如果顶部检查不够）
                 st.session_state.output_text = "生成思维导图失败：缺少 Graphviz 依赖。"
                 log_messages.append("❌ **生成失败：缺少 Graphviz 依赖。**")
                 status_placeholder.error("\n".join(log_messages))
                 st.error("❌ 无法执行 Graphviz，请确保已正确安装并配置其系统路径。")
             except Exception as e:
                 st.session_state.output_text = f"生成思维导图失败: {e}"
                 log_messages.append(f"❌ **生成思维导图时出错: {e}**")
                 log_messages.append(traceback.format_exc())
                 status_placeholder.error("\n".join(log_messages))
                 st.error(f"❌ 生成思维导图时出错: {e}")

# --- 主输出区域 和 问答 (QA) 区域 (保持不变) ---
st.markdown("---")
st.markdown("###  输出结果 / 视频问答")
output_text_area = st.text_area("结果展示区", value=st.session_state.get('output_text', ''), height=250, key="output_area")
qa_enabled = st.session_state.get('video_context') is not None
question = st.text_input("请输入你关于视频内容的问题:", placeholder="例如：视频主要讨论了哪些议题？", disabled=not qa_enabled, key="qa_input", help="请先点击“视频总结与问答准备”按钮生成视频理解上下文后，再进行提问。" if not qa_enabled else "")

if st.button("💡 提交问题进行问答", disabled=not qa_enabled, key="btn_qa"):
    if question.strip():
        status_placeholder.info("🤖 **请求大模型回答中...**")
        try:
            answer = ask_question_about_video(st.session_state.video_context, question, st.session_state.api_key)
            st.session_state.qa_answer = answer
            status_placeholder.empty()
        except Exception as e:
            st.session_state.qa_answer = f"问答时发生错误: {e}"
            status_placeholder.error(f"❌ 请求问答时发生错误: {e}")
            st.error(f"❌ 请求问答时发生错误: {e}")
    else:
        st.warning("⚠️ 请先在上面的输入框中输入你的问题。")

st.text_area("🧠 大模型回答区:", value=st.session_state.get('qa_answer', ''), height=150, key="qa_answer_area")

# --- 文件路径显示 (保持不变) ---
if st.session_state.video_dir:
    with st.expander("📂 查看已生成文件的路径 (点击展开)"):
        path_map = {
            "视频文件": st.session_state.get('video_path'), "音频文件": st.session_state.get('audio_path'),
            "转录 JSON": st.session_state.get('json_path'), "清洗后文本": st.session_state.get('text_path'),
            "关键帧 JSON": st.session_state.get('keyframe_json_path'), "CLIP 特征": st.session_state.get('clip_features_path'),
            "总结 JSON": st.session_state.get('summary_path')
            # 可以动态添加思维导图路径
        }
        # 动态添加思维导图路径（如果生成了）
        mindmap_file = os.path.join(st.session_state.video_dir, f"{st.session_state.get('current_video_name', 'video')}_summary_mindmap.png")
        if os.path.exists(mindmap_file):
             path_map["思维导图 PNG"] = mindmap_file

        all_paths_found = True
        for label, path in path_map.items():
             if path and os.path.exists(path): st.markdown(f"- ✅ **{label}:** `{path}`")
             elif path:
                 st.markdown(f"- ❌ **{label}:** (文件未找到) `{path}`"); all_paths_found = False
        # if all_paths_found: st.markdown("_所有预期文件均已生成。_") # 这句话可能不准确，只检查了部分预期文件