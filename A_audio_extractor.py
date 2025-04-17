#提取音频

import subprocess

def extract_audio_from_video(video_path, audio_path):
    ffmpeg_cmd = [
        'ffmpeg', '-y',           # -y 覆盖输出文件
        '-i', video_path,        # 输入视频路径
        '-vn',                   # 不要视频
        '-acodec', 'libmp3lame', # 使用 mp3 编码
        audio_path               # 输出音频路径
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return audio_path
