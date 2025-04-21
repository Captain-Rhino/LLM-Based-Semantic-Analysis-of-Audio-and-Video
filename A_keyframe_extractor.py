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
import math


class KeyframeExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = load_from_name("ViT-B-16", device=device)
        self.model.eval()

    def extract_keyframes(self, video_path, output_dir, asr_data=None, audio_path=None):
        """
        三模式关键帧抽取：
        1. 文本引导（有语音时）
        2. 视觉补偿（静默区间）
        3. 纯视觉（无语音时）
        """
        # 检测静默区间
        silent_ranges = self._detect_silent_ranges(audio_path) if audio_path else []

        # 👇 自动补前段静默区域（如果存在）
        if asr_data and len(asr_data) > 0 and asr_data[0]["start"] > 0:
            print(f"🔍 检测到前段静默：0.0 ~ {asr_data[0]['start']} 秒，将自动补帧")
            silent_ranges.insert(0, (0.0, asr_data[0]["start"]))

        if asr_data and len(asr_data) > 0:
            if silent_ranges:
                return self._hybrid_extraction(video_path, output_dir, asr_data, silent_ranges)
            return self._text_guided_extraction(video_path, output_dir, asr_data)

        return self._visual_guided_extraction(video_path, output_dir)

    def _hybrid_extraction(self, video_path, output_dir, asr_data, silent_ranges):
        """混合模式抽取（文本引导 + 静默补帧）"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        processed_frames = set()

        # 文本引导抽取
        text_kf = self._text_guided_extraction(video_path, output_dir, asr_data)
        keyframes.extend(text_kf)
        processed_frames.update(kf["frame_idx"] for kf in text_kf)

        # 视觉补偿抽取（“每2秒抽1帧，向上取整”）
        for start, end in silent_ranges:
            duration = end - start
            num_frames = int(np.ceil(duration / 2))
            if num_frames == 0:
                continue

            interval = duration / num_frames  # 每帧间隔（秒）
            for i in range(num_frames):
                timestamp = start + i * interval
                frame_idx = int(timestamp * fps)

                # ✅ 防止重复抽帧
                if frame_idx in processed_frames:
                    continue
                processed_frames.add(frame_idx)

                # 抽帧并写入图像
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                save_path = os.path.join(output_dir, f"comp_kf_{frame_idx:05d}.jpg")
                cv2.imwrite(save_path, frame)

                keyframes.append({
                    "mode": "visual_compensate",
                    "frame_idx": frame_idx,
                    "frame_rank": len([k for k in keyframes if k["mode"] == "visual_compensate"]),
                    "timestamp": round(frame_idx / fps, 2),
                    "start": start,
                    "end": end,
                    "text": "",
                    "text_len": 0,
                    "importance": self._calc_frame_importance(frame),
                    "image_path": save_path
                })

        cap.release()
        return sorted(keyframes, key=lambda x: x["timestamp"])

    def _text_guided_extraction(self, video_path, output_dir, asr_data):
        """文本引导的关键帧抽取（含CLIP特征保存）"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []

        # 创建特征存储目录
        feat_dir = os.path.join(output_dir, "clip_features")
        os.makedirs(feat_dir, exist_ok=True)

        for seg_idx, seg in enumerate(tqdm(asr_data, desc="文本引导抽帧")):
            start_frame = int(seg["start"] * fps)
            end_frame = int(seg["end"] * fps)
            text = seg["text"]

            # 动态计算抽帧数量（根据文本长度）
            num_frames = max(1, min(30, len(text) // 20))  # 每段最多30帧
            step = max(1, (end_frame - start_frame) // (num_frames + 1))
            frame_indices = [start_frame + step * i for i in range(1, num_frames + 1)]

            # 文本特征提取（提前计算）
            text_input = torch.cat([clip.tokenize(text)]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text_input)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)

            for rank, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue

                # 图像特征提取
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_feat = self.model.encode_image(image_input)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    sim = (image_feat * text_feat).sum().item()

                # 保存关键帧图像
                frame_filename = f"text_kf_{seg_idx:03d}_{frame_idx:05d}.jpg"
                save_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(save_path, frame)

                # 保存CLIP特征（新增）
                feat_filename = f"feat_{seg_idx:03d}_{frame_idx:05d}.pt"
                torch.save({
                    "image_feat": image_feat.cpu(),
                    "text_feat": text_feat.cpu()
                }, os.path.join(feat_dir, feat_filename))

                keyframes.append({
                    "mode": "text_guided",
                    "segment_idx": seg_idx,
                    "frame_idx": frame_idx,
                    "frame_rank": rank,
                    "timestamp": round(frame_idx / fps, 2),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "text_len": len(text),
                    "similarity": round(sim, 4),
                    "image_path": save_path,
                    "feat_path": os.path.join(feat_dir, feat_filename)  # 新增特征路径
                })

        cap.release()

        # 合并所有特征到单个文件（适配训练）
        self._compile_features(feat_dir, os.path.join(output_dir, "clip_features.pth"))

        return keyframes

    def _compile_features(self, feat_dir, output_path):
        """将所有特征编译为训练用的.pth文件"""
        image_feats = []
        text_feats = []

        for feat_file in sorted(glob.glob(os.path.join(feat_dir, "*.pt"))):
            data = torch.load(feat_file)
            image_feats.append(data["image_feat"])
            text_feats.append(data["text_feat"])

        torch.save({
            "image_feats": torch.cat(image_feats),
            "text_feats": torch.cat(text_feats)
        }, output_path)

        print(f"✅ CLIP特征已编译保存至 {output_path}")

#原来的text_guided,不带_compile_features_
    # def _text_guided_extraction(self, video_path, output_dir, asr_data):
    #     """文本引导模式（保留所有字段）"""
    #     cap = cv2.VideoCapture(video_path)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     keyframes = []
    #
    #     for seg_idx, seg in enumerate(tqdm(asr_data, desc="文本引导抽帧")):
    #         start_frame = int(seg["start"] * fps)
    #         end_frame = int(seg["end"] * fps)
    #         text = seg["text"]
    #
    #         # 动态抽帧
    #         num_frames = max(1, len(text) // 20)
    #         step = max(1, (end_frame - start_frame) // (num_frames + 1))
    #         frame_indices = [start_frame + step * i for i in range(1, num_frames + 1)]
    #
    #         # 文本特征
    #         text_input = torch.cat([clip.tokenize(text)]).to(self.device)
    #         with torch.no_grad():
    #             text_feat = self.model.encode_text(text_input)
    #             text_feat /= text_feat.norm(dim=-1, keepdim=True)
    #
    #         for rank, frame_idx in enumerate(frame_indices):
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #             ret, frame = cap.read()
    #             if not ret: continue
    #
    #             # 计算相似度
    #             image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #             image_input = self.preprocess(image).unsqueeze(0).to(self.device)
    #             with torch.no_grad():
    #                 image_feat = self.model.encode_image(image_input)
    #                 image_feat /= image_feat.norm(dim=-1, keepdim=True)
    #                 sim = (image_feat * text_feat).sum().item()
    #
    #             # 保存结果（保留所有字段）
    #             save_path = os.path.join(output_dir, f"text_kf_{seg_idx:03d}_{frame_idx:05d}.jpg")
    #             cv2.imwrite(save_path, frame)
    #             keyframes.append({
    #                 "mode": "text_guided",
    #                 "segment_idx": seg_idx,
    #                 "frame_idx": frame_idx,
    #                 "frame_rank": rank,
    #                 "timestamp": round(frame_idx / fps, 2),
    #                 "start": seg["start"],
    #                 "end": seg["end"],
    #                 "text": text,
    #                 "text_len": len(text),
    #                 "similarity": round(sim, 4),
    #                 "image_path": save_path
    #             })
    #
    #     cap.release()
    #     return keyframes

    def _visual_guided_extraction(self, video_path, output_dir, num_frames=10):
        """纯视觉模式（新增字段）"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_scores = []

        # 计算帧重要性
        for i in tqdm(range(total_frames), desc="视觉引导抽帧"):
            ret, frame = cap.read()
            if not ret: break
            frame_scores.append((i, self._calc_frame_importance(frame)))

        # 均衡选取
        selected_indices = []
        step = total_frames // num_frames
        for i in range(num_frames):
            start = i * step
            end = (i + 1) * step
            candidates = [idx for idx, _ in frame_scores if start <= idx < end]
            if candidates: selected_indices.append(candidates[0])

        # 保存结果
        keyframes = []
        for rank, idx in enumerate(sorted(selected_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = cap.read()
            save_path = os.path.join(output_dir, f"visual_kf_{idx:05d}.jpg")
            cv2.imwrite(save_path, frame)
            keyframes.append({
                "mode": "visual_guided",
                "frame_idx": idx,
                "frame_rank": rank,
                "timestamp": round(idx / fps, 2),
                "start": idx / fps,
                "end": (idx + 1) / fps,
                "text": "",
                "text_len": 0,
                "importance": frame_scores[idx][1],
                "image_path": save_path
            })

        cap.release()
        return keyframes

    def _calc_frame_importance(self, frame):
        """视觉重要性计算（不变）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= hist.sum() + 1e-10
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            clip_score = features.norm().item()

        return 0.7 * entropy + 0.3 * clip_score

    def _detect_silent_ranges(self, audio_path, min_silence_duration=2.0):
        """
        基于 ASR JSON 文件 + 音频时长，推断静默区间（单位：秒）
        - min_silence_duration：判断静默的最小间隔（秒）
        """
        import json

        # 🔧 获取音频名（不带后缀）
        video_name = os.path.splitext(os.path.basename(audio_path))[0]

        # 🗂 构建默认输出目录路径
        output_dir = os.path.join(os.path.dirname(audio_path), f"CNCLIP_keyframes_{video_name}")
        json_path = os.path.join(output_dir, f"{video_name}.json")

        if not os.path.exists(json_path):
            print(f"❌ 未找到 ASR JSON 文件: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            asr_segments = json.load(f)

        # 📏 获取音频时长
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = len(y) / sr

        # 🧠 计算静默区间
        silence_ranges = []
        last_end = 0.0

        for seg in asr_segments:
            current_start = seg["start"]
            if current_start - last_end >= min_silence_duration:
                silence_ranges.append((last_end, current_start))
            last_end = seg["end"]

        if total_duration - last_end >= min_silence_duration:
            silence_ranges.append((last_end, total_duration))

        print(f"✅ 共检测到静默区间 {len(silence_ranges)} 段：{silence_ranges}")
        return silence_ranges
