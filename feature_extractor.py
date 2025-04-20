# feature_extractor.py
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
import soundfile as sf
import os

# --- VGGish 配置和加载 ---
# 尝试加载 VGGish 模型
# 注意：首次运行时会自动下载模型，需要网络连接
# 如果 torch.hub 加载失败，可能需要检查网络或手动下载模型文件
try:
    # 确保 cache 目录存在且可写，如果需要可以指定 TORCH_HOME 环境变量
    # os.environ['TORCH_HOME'] = '/path/to/your/cache' # 可选
    print("尝试加载 VGGish 模型...")
    vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish_model.eval() # 设置为评估模式
    print("VGGish 模型加载成功。")
    VGGISH_AVAILABLE = True
except Exception as e:
    print(f"❌ 错误：加载 VGGish 模型失败: {e}")
    print("后续音频特征提取将使用零向量代替。请检查网络连接或 torch hub 设置。")
    vggish_model = None
    VGGISH_AVAILABLE = False

TARGET_SR = 16000 # VGGish 需要 16kHz 单声道音频
# 全局重采样器，避免重复初始化
resampler_dict = {}

def get_resampler(orig_freq, new_freq):
    """获取或创建重采样器"""
    if (orig_freq, new_freq) not in resampler_dict:
        print(f"Initializing resampler from {orig_freq}Hz to {new_freq}Hz")
        resampler_dict[(orig_freq, new_freq)] = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler_dict[(orig_freq, new_freq)]

def extract_audio_features(audio_segment_np, sample_rate):
    """
    使用 VGGish 提取单个音频片段的特征 (128维向量)
    audio_segment_np: numpy array 形式的音频片段
    sample_rate: 音频片段的采样率
    """
    if not VGGISH_AVAILABLE or vggish_model is None:
        # print("警告: VGGish模型不可用，返回零向量。")
        return np.zeros(128) # 返回一个符合维度的零向量

    try:
        # 转换为 Tensor
        audio_tensor = torch.from_numpy(audio_segment_np).float()

        # 重采样到 16kHz
        if sample_rate != TARGET_SR:
            resampler = get_resampler(sample_rate, TARGET_SR)
            audio_tensor = resampler(audio_tensor)

        # 检查并调整维度以符合 VGGish 输入要求 (通常需要 [batch, channels, samples] 或类似)
        # torchvggish 似乎可以直接处理 [samples] 或 [batch, samples]
        if audio_tensor.ndim == 1:
             audio_tensor = audio_tensor.unsqueeze(0) # -> [1, samples]

        # 确保音频长度足够 (VGGish通常处理0.96s窗口，这里简化)
        # 如果片段太短，VGGish可能会报错或输出NaN，可以考虑padding或返回零
        min_len_samples = int(0.1 * TARGET_SR) # 示例：至少0.1秒
        if audio_tensor.shape[1] < min_len_samples:
            # print(f"警告: 音频片段过短 ({audio_tensor.shape[1]/TARGET_SR:.2f}s)，返回零向量。")
            return np.zeros(128)

        with torch.no_grad():
            # 调用 VGGish 模型获取 embedding
            embeddings = vggish_model.forward(audio_tensor, fs=TARGET_SR) # 使用 forward 方法

        # 返回 embedding (通常是 [batch_size, 128])
        # 确保返回 numpy 数组
        result = embeddings.squeeze().cpu().numpy()
        if np.isnan(result).any():
            # print("警告：VGGish 输出包含 NaN，返回零向量。")
            return np.zeros(128)
        return result

    except Exception as e:
        print(f"❌ 错误：VGGish 特征提取失败: {e}")
        # print(f"输入音频 shape: {audio_segment_np.shape}, 采样率: {sample_rate}")
        return np.zeros(128) # 出错时返回零向量


def get_audio_segment(audio_path, start_time, end_time):
    """从音频文件中加载指定时间段的numpy数组和采样率"""
    try:
        # 使用 soundfile 获取信息，减少 I/O
        info = sf.info(audio_path)
        sr = info.samplerate
        start_frame = int(start_time * sr)
        end_frame = int(end_time * sr)
        duration_frames = end_frame - start_frame

        if duration_frames <= 0:
            # print(f"警告: 请求的音频片段时长为零或负数 ({start_time}-{end_time}s)")
            return None, sr

        # 加载指定片段
        audio_segment, actual_sr = sf.read(audio_path, start=start_frame, frames=duration_frames, dtype='float32', always_2d=False)

        if actual_sr != sr:
             print(f"警告：soundfile 读取的采样率 ({actual_sr}) 与 info ({sr}) 不符！")
             sr = actual_sr # 以实际读取为准

        # 如果是多声道，转为单声道 (平均)
        if audio_segment.ndim > 1 and audio_segment.shape[1] > 1:
            audio_segment = np.mean(audio_segment, axis=1)

        return audio_segment, sr
    except Exception as e:
        print(f"❌ 错误：加载音频片段失败 {audio_path} ({start_time}-{end_time}s): {e}")
        # 尝试返回文件的原始采样率，即使加载失败
        try:
            info = sf.info(audio_path)
            return None, info.samplerate
        except:
             return None, 0 # 如果连 info 都获取失败