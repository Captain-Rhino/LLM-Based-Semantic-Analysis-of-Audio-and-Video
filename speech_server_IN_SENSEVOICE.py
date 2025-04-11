# server.py

import time
import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile as sf
from io import BytesIO
import json

# FastAPI 实例
app = FastAPI()

# ASR Model
model_dir = "iic/SenseVoiceSmall"
vad_model_dir = "fsmn-vad"  # VAD模型路径

# 加载VAD模型
vad_model = AutoModel(
    model=vad_model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    disable_update=True
)

# 加载SenseVoice模型
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    disable_update=True
)

# ASR 输入模型
class ASRItem(BaseModel):
    wav: str  # 输入音频，base64 编码的字符串格式

@app.post("/asr")
async def asr(item: ASRItem):
    try:
        # 解码输入的 base64 编码音频数据并保存为 .wav 文件
        audio_data = base64.b64decode(item.wav)
        with open("test.wav", "wb") as f:
            f.write(audio_data)

        # 加载原始音频数据
        audio_data, sample_rate = sf.read("test.wav")

        # 使用VAD模型处理音频文件
        vad_res = vad_model.generate(
            input="test.wav",
            cache={},
            max_single_segment_time=30000,  # 最大单个片段时长
        )

        # 从VAD模型的输出中提取每个语音片段的开始和结束时间
        segments = vad_res[0]['value']

        # 对每个语音片段进行处理
        results = []
        for segment in segments:
            start_time, end_time = segment  # 获取开始和结束时间
            cropped_audio = audio_data[int(start_time * sample_rate / 1000): int(end_time * sample_rate / 1000)]

            # 将裁剪后的音频保存为临时文件
            temp_audio_file = "temp_cropped.wav"
            sf.write(temp_audio_file, cropped_audio, sample_rate)

            # 语音转文字处理
            res = model.generate(
                input=temp_audio_file,
                cache={},
                language="auto",  # 自动检测语言
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,  # 启用 VAD 断句
                merge_length_s=10000,  # 合并长度，单位为毫秒
            )

            # 处理输出结果
            text = rich_transcription_postprocess(res[0]["text"])

            # 将时间戳和文本内容保存到结果列表
            results.append({
                "start": start_time / 1000.0,  # 转换为秒
                "end": end_time / 1000.0,  # 转换为秒
                "text": text
            })

        return {"code": 0, "msg": "ok", "res": results}
    except Exception as e:
        return {"code": 1, "msg": str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2002)  # 启动 FastAPI 服务器
