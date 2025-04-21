from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import numpy as np
from PIL import Image
import os
import json

def load_stopwords(file_path):
    """
    从txt文件加载停用词列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def generate_wordcloud(transcription, summary_path, output_dir, video_name):
    """
    结合语音转录和视频总结生成词云
    :param transcription: 语音识别结果列表
    :param summary_path: 视频总结JSON文件路径
    :param output_dir: 输出目录
    :param video_name: 视频名称
    """
    # 1. 加载总结文本
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
        summary_text = summary_data.get("summary", "")

    # 2. 合并所有文本（转录+总结）
    transcript_text = " ".join([seg["text"] for seg in transcription if seg.get("text")])
    combined_text = f"{transcript_text} {summary_text}"

    #3. 停用词路径
    stopwords_path = r"G:\videochat\my_design\baidu_stopwords.txt"
    # 过滤停用词和单字
    stopwords = load_stopwords(stopwords_path) # 补充常用停用词

    # 4. 高级文本处理
    words = []
    for word in jieba.lcut(combined_text):
        if len(word) > 1 and word not in stopwords and not word.isspace():
            # 处理特殊符号和emoji
            clean_word = ''.join(c for c in word if c.isalnum() or c in ['#', '@'])
            if clean_word:
                words.append(clean_word)

    # 4. 词频统计（带权重）
    word_freq = Counter(words)

    # 5. 生成词云（带样式优化）
    wc = WordCloud(
        font_path="msyh.ttc",  # 微软雅黑
        width=1200,
        height=800,
        background_color="white",
        colormap="viridis",  # 使用科学配色
        max_words=200,
        collocations=False,  # 避免词组重复
        prefer_horizontal=0.8,  # 横向文字比例
        mask=np.array(Image.open("cloud_mask.png")) if os.path.exists("cloud_mask.png") else None
    ).generate_from_frequencies(word_freq)

    # 6. 保存结果
    output_path = os.path.join(output_dir, f"{video_name}_combined_wordcloud.png")
    wc.to_file(output_path)

    # 同时保存处理后的文本和词频
    # with open(os.path.join(output_dir, f"{video_name}_processed_text.txt"), "w", encoding="utf-8") as f:
    #     f.write(combined_text)

    with open(os.path.join(output_dir, f"{video_name}_word_freq.json"), "w", encoding="utf-8") as f:
        json.dump(word_freq, f, ensure_ascii=False, indent=2)

    print(f"✅ 综合词云已生成：{output_path}")
    return output_path



from graphviz import Digraph
import os
import json
import re

def generate_mindmap_from_summary(summary_path, output_dir, video_name):
    """
    从模型总结 summary.json 文件中提取结构层级，生成中文思维导图
    """
    # 读取总结文件
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
        summary_text = summary_data.get("summary", "").strip()

    if not summary_text:
        print("❌ 总结为空，无法生成思维导图")
        return None

    # 正则提取结构：每项以编号 1. / 2. / 3. 开头
    pattern = r"\d+\.\s+\*\*(.*?)\*\*：([\s\S]*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, summary_text)

    dot = Digraph(comment='Mindmap_from_summary', format='png')
    dot.attr('node', shape='box', fontname="Microsoft YaHei")
    center = f"{video_name}_主题"
    dot.node(center)

    for title, content in matches:
        dot.node(title)
        dot.edge(center, title)

        # 提取建议方向子点（如：- 建议方向：xxx）
        sub_items = re.findall(r"[*\-•]\s*(.*?)\n", content)
        for item in sub_items:
            clean_item = item.strip(" \n\t\r:：")
            if clean_item:
                dot.node(clean_item)
                dot.edge(title, clean_item)

    # 保存图像
    output_path = os.path.join(output_dir, f"{video_name}_summary_mindmap")
    dot.render(output_path, cleanup=True)

    print(f"🧠 基于总结的思维导图已生成：{output_path}.png")
    return output_path + ".png"
