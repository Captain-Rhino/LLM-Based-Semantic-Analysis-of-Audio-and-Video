# --- START OF FILE A_keyframe_extractor.py ---

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from cn_clip.clip import load_from_name
import cn_clip.clip as clip
import librosa
import glob
import math # 导入 math 模块用于 ceil
import json

class KeyframeExtractor:
    def __init__(self, device="cuda"):
        """
        初始化关键帧提取器。
        Args:
            device (str): 使用的设备 ('cuda' 或 'cpu')。
        """
        self.device = device
        print(f"关键帧提取器：正在加载 CLIP 模型到 {device}...")
        # 加载中文 CLIP 模型和预处理器
        self.model, self.preprocess = load_from_name("ViT-B-16", device=device)
        self.model.eval() # 设置为评估模式
        print("关键帧提取器：CLIP 模型加载完成。")

    # === 修改点 1: 添加 frame_interval 和 text_threshold 参数 ===
    def extract_keyframes(self, video_path, output_dir, asr_data=None, audio_path=None, frame_interval=2, text_threshold=80):
        """
        主函数，根据是否有语音数据选择不同的关键帧抽取策略。
        新增参数以支持自定义抽帧间隔和文本长度阈值。

        Args:
            video_path (str): 输入视频文件的路径。
            output_dir (str): 保存关键帧图像和特征的目录。
            asr_data (list, optional): 语音识别结果列表，包含'start', 'end', 'text'。默认为 None。
            audio_path (str, optional): 音频文件路径，用于检测静默。默认为 None。
            frame_interval (int): 视觉抽帧的时间间隔（秒）。默认为 2。
            text_threshold (int): 文本引导抽帧时，每多少个字符大约抽一帧的阈值。默认为 80。

        Returns:
            list: 包含所有提取出的关键帧信息的列表，按时间戳排序。
        """
        print(f"开始关键帧提取：视频='{video_path}', 输出目录='{output_dir}'")
        print(f"抽帧参数：视觉间隔={frame_interval}s, 文本阈值={text_threshold}字/帧")

        # --- 静默区间检测 ---
        silent_ranges = []
        if audio_path and os.path.exists(audio_path):
            print(f"检测音频文件 '{audio_path}' 中的静默区间...")
            # === 修改点 2: 将 frame_interval 传递给 _detect_silent_ranges ===
            # 注意：原_detect_silent_ranges实现不依赖frame_interval，这里保持原样，
            # 但如果需要基于间隔调整静默检测，可以在这里传递。
            # 实际使用 frame_interval 的地方在 _hybrid_extraction 的视觉补偿部分。
            silent_ranges = self._detect_silent_ranges(audio_path, output_dir) # 传递 output_dir 以查找 ASR JSON
        else:
            print("警告：未提供有效音频路径，无法检测静默区间。")

        # --- 处理视频开头的静默 ---
        # 如果有ASR数据，且第一段语音不是从0秒开始，则认为开头是静默
        if asr_data and len(asr_data) > 0 and asr_data[0].get("start", 0) > 0.1: # 加个小阈值避免0.0几秒的误差
            start_silence_end = asr_data[0]["start"]
            print(f"检测到视频开头存在静默：0.0 ~ {start_silence_end:.2f} 秒，将尝试补充视觉帧。")
            # 将开头的静默区间加入列表的最前面
            silent_ranges.insert(0, (0.0, start_silence_end))
            # 去重，以防万一检测函数也检测到了完全一样的区间
            silent_ranges = sorted(list(set(silent_ranges)))

        # --- 根据 ASR 数据选择提取模式 ---
        if asr_data and len(asr_data) > 0:
            # 如果有语音数据
            if silent_ranges:
                # 如果同时检测到静默区间，使用混合模式
                print("检测到语音和静默区间，使用混合模式抽取...")
                # === 修改点 3: 将参数传递给 _hybrid_extraction ===
                return self._hybrid_extraction(video_path, output_dir, asr_data, silent_ranges, frame_interval, text_threshold)
            else:
                # 如果只有语音数据，使用纯文本引导模式
                print("只有语音数据，使用文本引导模式抽取...")
                # === 修改点 4: 将参数传递给 _text_guided_extraction ===
                return self._text_guided_extraction(video_path, output_dir, asr_data, text_threshold)
        else:
            # 如果没有语音数据，使用纯视觉模式
            print("没有检测到语音数据，使用纯视觉模式抽取...")
            # === 修改点 5: 将参数传递给 _visual_guided_extraction ===
            return self._visual_guided_extraction(video_path, output_dir, frame_interval)

    # === 修改点 6: 添加 frame_interval, text_threshold 参数 ===
    def _hybrid_extraction(self, video_path, output_dir, asr_data, silent_ranges, frame_interval, text_threshold):
        """
        混合模式抽取：结合文本引导抽帧和静默区间视觉补偿抽帧。

        Args:
            video_path (str): 视频路径。
            output_dir (str): 输出目录。
            asr_data (list): ASR 结果。
            silent_ranges (list): 静默时间区间列表 [(start1, end1), ...]。
            frame_interval (int): 视觉补偿抽帧的时间间隔。
            text_threshold (int): 文本引导抽帧的阈值。

        Returns:
            list: 合并并排序后的关键帧列表。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"警告：无法获取视频 FPS，将使用默认值 30。")
            fps = 30 # 设置一个默认值以防万一

        keyframes = []          # 存储所有关键帧信息
        processed_frames = set() # 用于记录已处理的帧索引，防止重复添加

        # --- 1. 文本引导抽帧 ---
        print("混合模式：执行文本引导抽帧...")
        # === 修改点 7: 将 text_threshold 传递给 _text_guided_extraction ===
        text_kf = self._text_guided_extraction(video_path, output_dir, asr_data, text_threshold)
        keyframes.extend(text_kf)
        # 将文本引导抽取的帧加入已处理集合
        processed_frames.update(kf["frame_idx"] for kf in text_kf)
        print(f"混合模式：文本引导抽帧完成，共 {len(text_kf)} 帧。")

        # --- 2. 静默区间视觉补偿抽帧 ---
        print(f"混合模式：对 {len(silent_ranges)} 个静默区间进行视觉补偿抽帧 (间隔约 {frame_interval} 秒)...")
        visual_comp_count = 0
        for start, end in silent_ranges:
            duration = end - start
            # 如果区间时长小于抽帧间隔的一半，可能不需要抽帧，或者至少抽一帧？这里先跳过太短的
            if duration < frame_interval / 2 and duration > 0.1:
                 print(f"   静默区间 ({start:.2f}s - {end:.2f}s) 时长过短，抽取 1 帧作为补偿。")
                 num_frames_to_extract = 1
            elif duration <= 0.1: # 忽略几乎为0的区间
                 continue
            else:
                 # === 修改点 8: 使用 frame_interval 计算抽帧数量 ===
                 # 每 frame_interval 秒抽取一帧，向上取整
                 num_frames_to_extract = int(math.ceil(duration / frame_interval))

            print(f"   区间 ({start:.2f}s - {end:.2f}s, 时长 {duration:.2f}s): 计划补偿 {num_frames_to_extract} 帧。")

            if num_frames_to_extract == 0:
                continue

            # 计算实际的抽帧时间点，尽量均匀分布在区间内
            # 如果只抽一帧，放在区间中间
            if num_frames_to_extract == 1:
                timestamps = [start + duration / 2]
            else:
                # 如果抽多帧，包含首尾附近（加一点偏移避免完全在边界）
                time_step = duration / num_frames_to_extract
                timestamps = [start + time_step * (i + 0.5) for i in range(num_frames_to_extract)]


            for timestamp in timestamps:
                frame_idx = int(timestamp * fps) # 计算帧索引

                # 检查该帧是否已被文本引导模式处理过
                if frame_idx in processed_frames:
                    print(f"      跳过已处理的帧: {frame_idx} (时间戳: {timestamp:.2f}s)")
                    continue
                processed_frames.add(frame_idx) # 加入已处理集合

                # 定位到指定帧并读取
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"      警告：无法读取帧索引 {frame_idx}。")
                    continue

                # --- 保存视觉补偿帧图像 ---
                # 文件名包含模式、帧索引
                save_path = os.path.join(output_dir, f"comp_kf_{frame_idx:06d}.jpg") # 使用6位帧索引
                try:
                    cv2.imwrite(save_path, frame)
                except Exception as e_write:
                    print(f"      错误：保存补偿帧图像到 '{save_path}' 失败: {e_write}")
                    continue # 保存失败则跳过此帧

                visual_comp_count += 1
                # --- 添加补偿帧信息到列表 ---
                keyframes.append({
                    "mode": "visual_compensate", # 模式：视觉补偿
                    "frame_idx": frame_idx,      # 帧索引
                    # "frame_rank": visual_comp_count, # 在补偿帧中的序号 (可选)
                    "timestamp": round(timestamp, 2), # 时间戳
                    "start": round(start, 2),     # 静默区间开始时间
                    "end": round(end, 2),         # 静默区间结束时间
                    "text": "",                   # 无对应文本
                    "text_len": 0,
                    "importance": self._calc_frame_importance(frame), # 计算视觉重要性 (可选)
                    "image_path": save_path       # 图像保存路径
                    # 注意：视觉补偿帧通常不计算 CLIP 特征和相似度，也没有 feat_path
                })

        print(f"混合模式：视觉补偿抽帧完成，共新增 {visual_comp_count} 帧。")
        cap.release() # 释放视频捕获对象

        # 按时间戳对所有关键帧进行排序
        return sorted(keyframes, key=lambda x: x["timestamp"])

    # === 修改点 9: 添加 text_threshold 参数 ===
    def _text_guided_extraction(self, video_path, output_dir, asr_data, text_threshold):
        """
        文本引导的关键帧抽取（包含CLIP特征保存）。

        Args:
            video_path (str): 视频路径。
            output_dir (str): 输出目录。
            asr_data (list): ASR 结果。
            text_threshold (int): 每多少字符抽一帧的阈值。

        Returns:
            list: 文本引导的关键帧列表。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30

        keyframes = [] # 存储关键帧信息
        all_feature_data = [] # 临时存储所有特征数据，用于最后编译

        # --- 创建用于存储单个特征文件的子目录 ---
        feat_dir = os.path.join(output_dir, "clip_features_individual") # 临时目录
        os.makedirs(feat_dir, exist_ok=True)

        # 遍历 ASR 数据中的每一段语音
        for seg_idx, seg in enumerate(tqdm(asr_data, desc="文本引导抽帧")):
            start_time = seg.get("start", 0)
            end_time = seg.get("end", start_time) # 如果没有结束时间，则认为与开始时间相同
            text = seg.get("text", "").strip() # 获取文本并去除首尾空格

            if not text: # 如果文本为空，跳过这一段
                print(f"   警告：语音段 {seg_idx} 文本为空，跳过。")
                continue
            if end_time <= start_time: # 如果时间无效，跳过
                print(f"   警告：语音段 {seg_idx} 时间无效 (start={start_time}, end={end_time})，跳过。")
                continue

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            text_len = len(text)

            # --- 动态计算本段语音需要抽取的帧数 ---
            # === 修改点 10: 使用 text_threshold 计算抽帧数量 ===
            # 基本逻辑：文本越长，抽的帧越多，但有上限
            # 如果 text_threshold > 0，则按阈值计算，否则至少抽1帧
            if text_threshold > 0:
                num_frames = int(math.ceil(text_len / text_threshold))
            else:
                num_frames = 1 # 如果阈值为0或负数，则每段至少抽1帧
            num_frames = max(1, min(30, num_frames)) # 确保至少1帧，最多30帧

            # --- 计算抽帧位置 ---
            # 在语音段的持续时间内均匀选取 num_frames 个点
            duration_frames = end_frame - start_frame
            if duration_frames <= 0 or num_frames == 0: # 如果持续时间为0或不需要抽帧
                 frame_indices = [start_frame] if duration_frames >= 0 else [] # 至少在起始帧抽一次（如果时间有效）
            elif num_frames == 1: # 如果只抽一帧，选中间位置
                 frame_indices = [start_frame + duration_frames // 2]
            else:
                 # 如果抽多帧，均匀分布（包含首尾附近）
                 step = duration_frames / num_frames
                 # 在每个子区间的中点取帧
                 frame_indices = [int(start_frame + step * (i + 0.5)) for i in range(num_frames)]


            # --- 提取文本特征 (对整段文本只计算一次) ---
            try:
                # 使用 clip.tokenize 处理文本
                text_input = clip.tokenize([text]).to(self.device) # 注意 tokenize 需要列表输入
                with torch.no_grad(): # 关闭梯度计算
                    # 计算文本的 CLIP 特征向量
                    text_feat = self.model.encode_text(text_input)
                    # L2 归一化，使其成为单位向量
                    text_feat /= text_feat.norm(dim=-1, keepdim=True)
            except Exception as e_text:
                print(f"   错误：计算文本段 {seg_idx} 的 CLIP 特征失败: {e_text}")
                continue # 如果文本特征提取失败，跳过这一整段语音

            # --- 遍历计算出的帧索引，提取图像特征并保存 ---
            for rank, frame_idx in enumerate(frame_indices):
                # 定位到视频帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read() # 读取帧图像
                if not ret:
                    print(f"      警告：无法读取帧索引 {frame_idx} (文本段 {seg_idx})。")
                    continue # 读取失败则跳过

                # --- 提取图像特征 ---
                try:
                    # 将 OpenCV 图像 (BGR) 转换为 PIL 图像 (RGB)
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # 使用 CLIP 的预处理器处理图像
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        # 计算图像的 CLIP 特征向量
                        image_feat = self.model.encode_image(image_input)
                        # L2 归一化
                        image_feat /= image_feat.norm(dim=-1, keepdim=True)
                        # 计算图像特征和文本特征的余弦相似度 (点积)
                        sim = (image_feat @ text_feat.T).item() # 使用矩阵乘法并取值
                except Exception as e_img:
                    print(f"      错误：处理帧 {frame_idx} (文本段 {seg_idx}) 的图像或计算特征失败: {e_img}")
                    continue # 图像处理失败则跳过

                # --- 保存关键帧图像 ---
                # 文件名包含模式、语音段索引、帧索引
                frame_filename = f"text_kf_{seg_idx:03d}_{frame_idx:06d}.jpg"
                save_path = os.path.join(output_dir, frame_filename)
                try:
                    cv2.imwrite(save_path, frame)
                except Exception as e_write_img:
                     print(f"      错误：保存关键帧图像到 '{save_path}' 失败: {e_write_img}")
                     continue # 保存失败跳过

                # --- 保存单个特征文件 (图像特征 + 对应文本特征) ---
                feat_filename = f"feat_{seg_idx:03d}_{frame_idx:06d}.pt"
                feat_save_path = os.path.join(feat_dir, feat_filename)
                try:
                    # 将特征移动到 CPU 保存，减小 GPU 显存占用
                    torch.save({
                        "image_feat": image_feat.cpu(), # 保存图像特征
                        "text_feat": text_feat.cpu()    # 保存对应的文本特征
                    }, feat_save_path)
                    # 记录特征信息，用于后续编译
                    all_feature_data.append({"image_feat_path": feat_save_path, "text": text})
                except Exception as e_write_feat:
                    print(f"      错误：保存特征文件到 '{feat_save_path}' 失败: {e_write_feat}")
                    # 特征保存失败，后续将无法使用此帧进行训练，但关键帧信息仍可记录
                    feat_save_path = None # 标记特征路径无效


                # --- 添加关键帧信息到列表 ---
                keyframes.append({
                    "mode": "text_guided",        # 模式：文本引导
                    "segment_idx": seg_idx,       # 对应的语音段索引
                    "frame_idx": frame_idx,       # 帧索引
                    "frame_rank": rank,           # 在该语音段内的抽帧序号
                    "timestamp": round(frame_idx / fps, 2), # 时间戳
                    "start": round(start_time, 2),# 语音段开始时间
                    "end": round(end_time, 2),    # 语音段结束时间
                    "text": text,                 # 对应的语音文本
                    "text_len": text_len,         # 文本长度
                    "similarity": round(sim, 4),  # 图文相似度
                    "image_path": save_path,      # 关键帧图像路径
                    "feat_path": feat_save_path   # 对应的特征文件路径 (可能为 None)
                })

        cap.release() # 释放视频对象

        # --- 编译所有保存的单个特征文件到一个 .pth 文件 ---
        # 这是为了方便后续的 CLIP Adaptor 微调训练
        compiled_feat_path = os.path.join(output_dir, "clip_features.pth") # 最终编译文件的路径
        self._compile_features(feat_dir, compiled_feat_path)

        # (可选) 清理临时的单个特征文件目录
        # import shutil
        # try:
        #     shutil.rmtree(feat_dir)
        #     print(f"   临时特征目录 {feat_dir} 已清理。")
        # except Exception as e_clean:
        #     print(f"   警告：清理临时特征目录 {feat_dir} 失败: {e_clean}")


        # 返回文本引导的关键帧列表
        return keyframes

    def _compile_features(self, feat_dir, output_path):
        """
        将指定目录下所有单独保存的 .pt 特征文件编译成一个 .pth 文件。
        这个 .pth 文件通常包含两个 tensors: 'image_feats' 和 'text_feats'，
        用于后续的 CLIP Adaptor 训练。

        Args:
            feat_dir (str): 包含单个 .pt 特征文件的目录。
            output_path (str): 编译后的 .pth 文件保存路径。
        """
        image_feats = [] # 存储所有图像特征
        text_feats = []  # 存储所有文本特征

        # 查找 feat_dir 下所有的 .pt 文件
        feature_files = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))

        if not feature_files:
            print(f"警告：在目录 '{feat_dir}' 中未找到任何 .pt 特征文件，无法编译。")
            return

        print(f"开始编译 {len(feature_files)} 个特征文件到 {output_path}...")
        # 遍历每个找到的 .pt 文件
        for feat_file in tqdm(feature_files, desc="编译特征"):
            try:
                # 加载单个特征文件的数据
                data = torch.load(feat_file, map_location='cpu') # 加载到 CPU
                # 检查需要的数据是否存在且是 Tensor
                if "image_feat" in data and isinstance(data["image_feat"], torch.Tensor) and \
                   "text_feat" in data and isinstance(data["text_feat"], torch.Tensor):
                    image_feats.append(data["image_feat"]) # 添加到图像特征列表
                    text_feats.append(data["text_feat"])   # 添加到文本特征列表
                else:
                    print(f"   警告：跳过文件 '{feat_file}'，缺少 'image_feat' 或 'text_feat' 或格式不正确。")
            except Exception as e_load:
                print(f"   错误：加载特征文件 '{feat_file}' 失败: {e_load}")

        # 检查是否成功加载了任何特征
        if not image_feats or not text_feats:
            print("错误：未能成功加载任何有效的图文特征对，编译中止。")
            return

        # --- 将特征列表堆叠成一个大的 Tensor ---
        # torch.cat 沿着第 0 维（batch 维）拼接
        try:
            compiled_data = {
                # image_feats 列表中的每个元素形状通常是 (1, embedding_dim)
                # cat 后形状变为 (N, embedding_dim)，N 是特征对数量
                "image_feats": torch.cat(image_feats, dim=0),
                # text_feats 列表中的每个元素形状通常也是 (1, embedding_dim)
                # cat 后形状变为 (N, embedding_dim)
                "text_feats": torch.cat(text_feats, dim=0)
            }
            # 保存编译后的数据到指定的 .pth 文件
            torch.save(compiled_data, output_path)
            print(f"✅ CLIP 特征已成功编译并保存至 {output_path}")
            print(f"   编译后的图像特征形状: {compiled_data['image_feats'].shape}")
            print(f"   编译后的文本特征形状: {compiled_data['text_feats'].shape}")

        except Exception as e_cat_save:
            print(f"错误：合并或保存编译后的特征时出错: {e_cat_save}")


    # === 修改点 11: 添加 frame_interval 参数 ===
    def _visual_guided_extraction(self, video_path, output_dir, frame_interval):
        """
        纯视觉引导的关键帧抽取模式。
        根据设定的时间间隔（frame_interval）抽取视觉上重要的帧。

        Args:
            video_path (str): 视频路径。
            output_dir (str): 输出目录。
            frame_interval (int): 每隔多少秒抽取一帧。

        Returns:
            list: 纯视觉关键帧列表。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取总帧数
        if total_frames <= 0:
             print(f"错误：无法获取视频总帧数或视频为空 {video_path}")
             cap.release()
             return []
        total_duration = total_frames / fps # 计算视频总时长（秒）

        # === 修改点 12: 使用 frame_interval 计算需要抽取的帧数 ===
        # 确保 frame_interval 大于 0，避免除零错误
        if frame_interval <= 0:
            print("警告：视觉抽帧间隔必须大于 0，将使用默认值 2 秒。")
            frame_interval = 2
        # 计算需要抽取的总帧数，向上取整
        num_frames_to_extract = int(math.ceil(total_duration / frame_interval))
        if num_frames_to_extract == 0 and total_duration > 0: # 避免时长>0但抽帧数为0的情况
             num_frames_to_extract = 1 # 至少抽一帧

        print(f"纯视觉模式：视频时长 {total_duration:.2f}s, 帧率 {fps:.2f}, 总帧数 {total_frames}")
        print(f"             计划每隔 {frame_interval}s 抽一帧，总共抽取 {num_frames_to_extract} 帧。")

        # --- 计算每一帧的视觉重要性得分 ---
        frame_scores = [] # 存储 (帧索引, 重要性得分)
        # 使用 tqdm 显示进度条
        for i in tqdm(range(total_frames), desc="计算视觉重要性"):
            ret, frame = cap.read()
            if not ret:
                print(f"警告：在计算重要性时无法读取帧 {i}。")
                break # 读取失败则停止
            # 计算当前帧的重要性得分
            importance = self._calc_frame_importance(frame)
            frame_scores.append((i, importance)) # 记录帧索引和得分

        # --- 均衡选取关键帧 ---
        # 目标是从整个视频中均匀地选出 num_frames_to_extract 帧，优先选重要性高的
        selected_indices = [] # 存储最终选定的帧索引
        if num_frames_to_extract > 0 and frame_scores:
            # 将视频大致分成 num_frames_to_extract 个时间段（或帧段）
            step = total_frames / num_frames_to_extract # 每个段大致包含的帧数
            for i in range(num_frames_to_extract):
                # 计算当前段的起始和结束帧索引
                start_idx = int(i * step)
                end_idx = int((i + 1) * step)
                # 从 frame_scores 中筛选出属于当前段的帧及其得分
                segment_scores = [(idx, score) for idx, score in frame_scores if start_idx <= idx < end_idx]

                if segment_scores:
                    # 如果当前段中有帧，则选择该段中重要性得分最高的帧
                    best_frame_in_segment = max(segment_scores, key=lambda item: item[1])
                    selected_indices.append(best_frame_in_segment[0]) # 只记录选中的帧索引
                # else: # 如果某个段恰好没有帧（可能发生在视频结尾），则跳过

        print(f"纯视觉模式：根据重要性均衡选取了 {len(selected_indices)} 帧。")

        # --- 保存选中的关键帧图像和信息 ---
        keyframes = []
        # 对选中的帧索引进行排序，确保输出是按时间顺序的
        for rank, frame_idx in enumerate(sorted(selected_indices)):
            # 重新定位到选中的帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"   警告：无法重新读取已选中的帧 {frame_idx}。")
                continue

            # 保存图像文件
            save_path = os.path.join(output_dir, f"visual_kf_{frame_idx:06d}.jpg") # 6位帧索引
            try:
                cv2.imwrite(save_path, frame)
            except Exception as e_write_img:
                 print(f"   错误：保存视觉关键帧图像到 '{save_path}' 失败: {e_write_img}")
                 continue

            # 查找该帧的重要性得分
            # frame_scores 是 (idx, score) 列表，可以直接用索引访问（如果没出错）
            # 或者更安全的方式是查找
            importance_score = 0.0
            for idx, score in frame_scores:
                 if idx == frame_idx:
                     importance_score = score
                     break

            # 添加关键帧信息
            keyframes.append({
                "mode": "visual_guided",      # 模式：纯视觉
                "frame_idx": frame_idx,       # 帧索引
                "frame_rank": rank,           # 在选出的视觉帧中的排名
                "timestamp": round(frame_idx / fps, 2), # 时间戳
                # "start": round(frame_idx / fps, 2), # (可选) 开始时间戳
                # "end": round((frame_idx + 1) / fps, 2), # (可选) 结束时间戳 (近似)
                "text": "",                   # 无对应文本
                "text_len": 0,
                "importance": round(importance_score, 4), # 视觉重要性得分
                "image_path": save_path       # 图像路径
                # 纯视觉模式通常不包含 similarity 和 feat_path
            })

        cap.release() # 释放视频对象
        print(f"纯视觉模式：抽帧完成，共保存 {len(keyframes)} 帧。")
        return keyframes

    def _calc_frame_importance(self, frame):
        """
        计算单帧图像的视觉重要性。
        结合了图像的灰度熵和 CLIP 特征的范数。

        Args:
            frame (numpy.ndarray): 输入的 BGR 图像帧。

        Returns:
            float: 计算出的重要性得分。
        """
        try:
            # --- 计算灰度熵 ---
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 计算灰度直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            # 归一化直方图
            hist /= (hist.sum() + 1e-10) # 加一个小数防止除以零
            # 计算熵 H = -sum(p*log2(p))
            entropy = -np.sum(hist * np.log2(hist + 1e-10)) # 加一个小数防止 log2(0)

            # --- 计算 CLIP 特征范数 ---
            # 将 BGR 转换为 RGB 并创建 PIL Image 对象
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 预处理图像并增加 batch 维度
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad(): # 无需计算梯度
                # 提取图像特征
                features = self.model.encode_image(image_input)
                # 计算特征向量的 L2 范数（长度）
                clip_score = features.norm().item() # .item() 获取标量值

            # --- 结合熵和 CLIP 分数 ---
            # 可以根据需要调整权重因子（这里是 0.7 和 0.3）
            importance = 0.7 * entropy + 0.3 * clip_score
            return round(importance, 4) # 保留4位小数

        except Exception as e:
            print(f"   错误：计算帧重要性时出错: {e}")
            return 0.0 # 出错时返回0分

    # === 修改点 13: 增加 output_dir 参数用于定位 ASR JSON ===
    def _detect_silent_ranges(self, audio_path, output_dir, min_silence_duration=2.0):
        """
        基于 ASR JSON 文件（如果存在）和音频总时长，推断静默区间。

        Args:
            audio_path (str): 音频文件路径。
            output_dir (str): 用于查找对应 ASR JSON 文件的目录。
            min_silence_duration (float): 判断为静默的最小时间间隔（秒）。默认为 2.0。

        Returns:
            list: 静默时间区间列表 [(start1, end1), (start2, end2), ...]，单位为秒。
                  如果 ASR 文件不存在或无法处理，返回空列表。
        """
        # --- 查找对应的 ASR JSON 文件 ---
        # 假设 ASR JSON 文件与音频文件同名（除了扩展名），且位于 output_dir 下
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        # 注意：这里修改了查找逻辑，不再依赖 output_dir 包含 "CNCLIP_keyframes_"
        # 而是直接在 output_dir 下查找 {audio_basename}.json 或 {audio_basename}_transcription.json
        json_path_option1 = os.path.join(output_dir, f"{audio_basename}.json")
        json_path_option2 = os.path.join(output_dir, f"{audio_basename}_transcription.json") # 兼容 Streamlit 的命名

        asr_json_path = None
        if os.path.exists(json_path_option1):
            asr_json_path = json_path_option1
        elif os.path.exists(json_path_option2):
            asr_json_path = json_path_option2

        if not asr_json_path:
            print(f"   信息：在 '{output_dir}' 目录下未找到对应的 ASR JSON 文件 "
                  f"('{os.path.basename(json_path_option1)}' 或 '{os.path.basename(json_path_option2)}')。"
                  " 无法基于 ASR 推断静默区间。")
            return []

        print(f"   找到 ASR JSON 文件: {asr_json_path}")
        try:
            # 读取 ASR JSON 文件
            with open(asr_json_path, "r", encoding="utf-8") as f:
                asr_segments = json.load(f)
            # 确保 asr_segments 是列表且包含 start/end
            if not isinstance(asr_segments, list) or not all('start' in seg and 'end' in seg for seg in asr_segments):
                 print(f"   错误：ASR JSON 文件 '{asr_json_path}' 格式不正确，缺少 'start' 或 'end' 键或不是列表。")
                 return []
        except json.JSONDecodeError:
            print(f"   错误：无法解析 ASR JSON 文件 '{asr_json_path}'。")
            return []
        except Exception as e:
            print(f"   错误：读取 ASR JSON 文件 '{asr_json_path}' 时出错: {e}")
            return []

        # --- 获取音频总时长 ---
        try:
            # 使用 librosa 加载音频获取时长
            y, sr = librosa.load(audio_path, sr=None) # sr=None 保留原始采样率
            total_duration = librosa.get_duration(y=y, sr=sr) # 获取时长（秒）
        except Exception as e:
            print(f"   错误：加载音频文件 '{audio_path}' 或获取时长失败: {e}")
            return [] # 无法获取时长，则无法判断结尾静默

        # --- 基于 ASR 时间戳计算静默区间 ---
        silence_ranges = [] # 存储检测到的静默区间
        last_end_time = 0.0 # 上一段语音的结束时间，初始为0

        # 对 ASR 段按开始时间排序，以防万一顺序是乱的
        asr_segments.sort(key=lambda x: x.get("start", 0))

        # 遍历排序后的语音段
        for seg in asr_segments:
            current_start_time = seg.get("start", 0)
            current_end_time = seg.get("end", current_start_time) # 若无end，则认为时长为0

            # 计算当前语音段开始时间与上一段结束时间之间的间隔
            silence_duration = current_start_time - last_end_time

            # 如果间隔大于等于最小静默时长，则认为这是一个静默区间
            if silence_duration >= min_silence_duration:
                # 添加静默区间 (上一段结束时间, 当前段开始时间)
                silence_ranges.append((round(last_end_time, 2), round(current_start_time, 2)))

            # 更新上一段的结束时间，为下一次迭代做准备
            # 取当前结束时间和上一段结束时间的最大值，处理可能的重叠段
            last_end_time = max(last_end_time, current_end_time)


        # --- 检查最后一段语音结束后到音频结尾是否也存在静默 ---
        final_silence_duration = total_duration - last_end_time
        if final_silence_duration >= min_silence_duration:
            # 添加结尾的静默区间
            silence_ranges.append((round(last_end_time, 2), round(total_duration, 2)))

        if silence_ranges:
            print(f"   ✅ 基于 ASR 和音频时长，共检测到静默区间 {len(silence_ranges)} 段：")
            for r in silence_ranges: print(f"      {r[0]:.2f}s - {r[1]:.2f}s")
        else:
            print(f"   信息：未检测到持续时间超过 {min_silence_duration} 秒的静默区间。")

        return silence_ranges