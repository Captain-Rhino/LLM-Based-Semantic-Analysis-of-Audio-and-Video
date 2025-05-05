# --- START OF FILE A_visualizer.py ---

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import numpy as np
from PIL import Image
import os
import json
# 导入 graphviz 和 re
from graphviz import Digraph
import re

# --- load_stopwords 和 generate_wordcloud 函数保持不变 ---

def load_stopwords(file_path):
    """
    从txt文件加载停用词列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"警告：停用词文件未找到 {file_path}，将不使用停用词。")
        return set() # 返回空集合

def generate_wordcloud(transcription, summary_path, output_dir, video_name):
    """
    结合语音转录和视频总结生成词云
    :param transcription: 语音识别结果列表 (应为列表)
    :param summary_path: 视频总结JSON文件路径
    :param output_dir: 输出目录
    :param video_name: 视频名称
    """
    summary_text = ""
    # 1. 加载总结文本 (增加文件存在检查)
    if not os.path.exists(summary_path):
        print(f"❌ 错误：总结文件未找到 {summary_path}，无法生成词云图。")
        return None
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            summary_text = summary_data.get("summary", "")
    except json.JSONDecodeError:
        print(f"❌ 错误：无法解析总结文件 {summary_path}。")
        return None
    except Exception as e:
        print(f"❌ 读取总结文件时出错: {e}")
        return None


    # 2. 合并所有文本（转录+总结）
    # 确保 transcription 是列表
    transcript_text = ""
    if isinstance(transcription, list):
        transcript_text = " ".join([seg.get("text", "") for seg in transcription if seg.get("text")])
    else:
        print("警告：传入的 transcription 不是列表格式，词云图中可能缺少转录文本。")

    combined_text = f"{transcript_text} {summary_text}".strip()
    if not combined_text:
        print("❌ 合并后的文本为空，无法生成词云图。")
        return None

    # 3. 停用词路径和加载
    stopwords_path = r"G:\videochat\my_design\baidu_stopwords.txt" # 确保路径正确
    stopwords = load_stopwords(stopwords_path)

    # 4. 文本处理与分词
    words = []
    try:
        # 使用精确模式分词
        for word in jieba.lcut(combined_text):
            # 过滤条件：长度大于1，不是停用词，不是纯空格
            if len(word) > 1 and word not in stopwords and not word.isspace():
                # 简单清理：只保留字母、数字和中文，移除其他符号
                # \u4e00-\u9fa5 是中文字符范围
                clean_word = ''.join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', word))
                if clean_word:
                    words.append(clean_word)
    except Exception as e:
        print(f"❌ 分词或文本处理时出错: {e}")
        return None # 出错则不继续

    if not words:
        print("❌ 没有提取到有效词语（可能都被过滤了），无法生成词云图。")
        return None

    # 5. 词频统计
    word_freq = Counter(words)

    # 6. 生成词云图
    font_path = "msyh.ttc" # 微软雅黑字体路径，确保存在或更换
    mask_path = "cloud_mask.png" # 可选的蒙版图片路径
    try:
        wc = WordCloud(
            font_path=font_path,
            width=1200,
            height=800,
            background_color="white",
            colormap="viridis",
            max_words=200,
            collocations=False,
            prefer_horizontal=0.9, # 稍微增加横向比例
            # 如果字体路径无效会报错，需要用户确认
            # 如果蒙版图片不存在则不使用
            mask=np.array(Image.open(mask_path)) if os.path.exists(mask_path) else None
        ).generate_from_frequencies(word_freq)
    except IOError:
         print(f"❌ 错误：无法加载字体文件 '{font_path}'。请确保字体文件存在或更换字体路径。")
         # 尝试使用默认字体（可能不支持中文）
         try:
             print("   尝试使用默认字体生成...")
             wc = WordCloud(
                 width=1200, height=800, background_color="white", colormap="viridis",
                 max_words=200, collocations=False, prefer_horizontal=0.9,
                 mask=np.array(Image.open(mask_path)) if os.path.exists(mask_path) else None
             ).generate_from_frequencies(word_freq)
         except Exception as e_fallback:
             print(f"❌ 使用默认字体生成词云图也失败了: {e_fallback}")
             return None
    except Exception as e:
        print(f"❌ 生成词云对象时出错: {e}")
        return None


    # 7. 保存结果
    output_filename = f"{video_name}_combined_wordcloud.png"
    output_path = os.path.join(output_dir, output_filename)
    try:
        wc.to_file(output_path)
    except Exception as e:
        print(f"❌ 保存词云图到 '{output_path}' 时出错: {e}")
        return None

    # 保存词频（可选）
    word_freq_path = os.path.join(output_dir, f"{video_name}_word_freq.json")
    try:
        with open(word_freq_path, "w", encoding="utf-8") as f:
            json.dump(word_freq, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"警告：保存词频文件到 '{word_freq_path}' 时出错: {e}")

    print(f"✅ 综合词云已生成：{output_path}")
    return output_path


# --- generate_mindmap_from_summary 函数修改版 ---

def generate_mindmap_from_summary(summary_path, output_dir, video_name):
    """
    从模型总结 summary.json 文件中提取结构层级，生成中文思维导图。
    (修改版：尝试更灵活地解析常见的列表格式)

    Args:
        summary_path (str): 总结 JSON 文件路径。
        output_dir (str): 思维导图图片输出目录。
        video_name (str): 视频名称，用于节点标签和文件名。

    Returns:
        str or None: 成功则返回生成的 PNG 图片路径，否则返回 None。
    """
    # --- 1. 读取并解析总结文件 ---
    summary_text = ""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            summary_text = summary_data.get("summary", "").strip() # 获取总结文本并去除首尾空格
    except FileNotFoundError:
        print(f"❌ 错误：总结文件未找到 {summary_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误：无法解析总结文件 {summary_path} (非有效 JSON)")
        return None
    except Exception as e:
        print(f"❌ 读取或解析总结文件时发生意外错误: {e}")
        return None

    if not summary_text:
        print("❌ 总结内容为空，无法生成思维导图。")
        return None

    print(f"ℹ️ 开始为 '{video_name}' 生成思维导图...")
    # 打印部分总结内容，方便调试正则表达式
    print(f"   原始总结文本 (片段供参考): {summary_text[:300]}...")

    # --- 2. 初始化 Graphviz 图 ---
    # 使用 Digraph 创建有向图，指定格式为 png
    dot = Digraph(comment=f'Mindmap_{video_name}', format='png')
    # 设置全局节点属性：形状为方框，字体为微软雅黑（确保系统支持），字号10
    # !! 重要：如果"Microsoft YaHei"无效，请替换为 'SimHei', 'FangSong', 'KaiTi' 等 !!
    dot.attr('node', shape='box', fontname="Microsoft YaHei", fontsize='10')
    # 设置全局边属性：字体和字号
    dot.attr('edge', fontname="Microsoft YaHei", fontsize='9')

    # --- 3. 创建中心主题节点 ---
    center_label = f"{video_name}\n(视频主题)" # 中心节点标签，换行显示更清晰
    center_node_id = "___center_node___" # 使用一个特殊且固定的 ID
    # 设置中心节点样式：椭圆、填充浅蓝色
    dot.node(center_node_id, center_label, shape='ellipse', style='filled', fillcolor='lightblue')

    # --- 4. 尝试解析总结文本以提取主要分支 ---
    # 修改后的正则表达式：匹配行首的常见列表标记（数字+.、数字+空格、数字+）、*+空格、-+空格、•+空格）
    # 它会捕获两个组：group(1) 是列表标记本身，group(2) 是标记后面的所有文本
    pattern = r"^\s*(\d+[\.\)]\s*|\*\s+|\•\s+|\-\s+)(.*)"
    lines = summary_text.splitlines() # 将总结文本按行分割成列表

    main_points_found = 0 # 计数器，用于生成唯一节点 ID
    last_main_node_id = center_node_id # 记录上一个主节点 ID，用于潜在的子节点连接（当前简化版未使用）

    # 遍历每一行
    for line in lines:
        line = line.strip() # 去除当前行的首尾空格
        if not line: # 跳过空行
            continue

        match = re.match(pattern, line, re.UNICODE) # 尝试匹配行首的列表标记
        if match:
            # 如果匹配成功，说明这可能是一个主要分支点
            main_points_found += 1
            # 提取列表标记后面的文本作为标题 (group 2)
            title = match.group(2).strip()
            # 对标题做一些清理，移除末尾可能存在的冒号或句号
            title = title.strip(" :：。.")

            if title: # 确保提取到的标题不为空
                # 为每个主节点创建一个唯一的 ID (防止标题重复导致节点合并)
                node_id = f"main_node_{main_points_found}"
                # 创建思维导图节点
                dot.node(node_id, title)
                # 将该节点连接到中心主题节点
                dot.edge(center_node_id, node_id)
                print(f"   - 添加主要分支: {title}")
                last_main_node_id = node_id # 更新最后的主节点 ID

        # (当前版本简化处理，不显式处理子项。如果需要，可以在这里添加逻辑：
        # 例如，检查行是否以更多空格开头，或不匹配主项模式，然后连接到 last_main_node_id)

    # --- 5. 处理无法解析结构的情况 ---
    if main_points_found == 0:
        # 如果遍历完所有行都没有找到任何匹配列表模式的行
        print("   ⚠️ 未能从总结中检测到明确的列表结构 (如 '1.', '*', '-'). 将整个总结文本作为一个分支。")
        # 将整个总结文本（或其一部分）作为一个单独的节点
        # 截断长文本，防止节点过大难以阅读
        display_text = summary_text[:250] + '...' if len(summary_text) > 250 else summary_text
        # 替换文本中的换行符为 Graphviz 能识别的换行符 `\n`，或者直接替换为空格
        # display_text = display_text.replace('\r\n', '\\n').replace('\n', '\\n') # 保留换行
        display_text = display_text.replace('\n', ' ').replace('\r', '') # 替换为空格

        node_id = "full_summary_node"
        dot.node(node_id, display_text, shape='plaintext') # 使用 plaintext 形状可能更适合长文本
        dot.edge(center_node_id, node_id) # 连接到中心节点

    # --- 6. 保存生成的思维导图图像 ---
    # 输出文件基础名（不含扩展名）
    output_base = os.path.join(output_dir, f"{video_name}_summary_mindmap")
    try:
        # 调用 Graphviz 的 render 方法生成文件
        # cleanup=True 会删除中间生成的 .gv 文件
        # view=False 表示不自动打开生成的图片
        rendered_path = dot.render(filename=output_base, format='png', cleanup=True, view=False)
        # dot.render 会自动在 output_base 后面加上 .png
        print(f"✅ 思维导图已成功生成并保存至: {rendered_path}")
        return rendered_path # 返回实际生成的 PNG 文件路径

    except FileNotFoundError as e_gv: # 捕获找不到 dot 命令的错误
        print(f"❌ 生成思维导图图像失败: 找不到 Graphviz 的 'dot' 命令。")
        print(f"   请确保已正确安装 Graphviz 软件本身，并将其 'bin' 目录添加至系统 PATH 环境变量。错误详情: {e_gv}")
        return None
    except Exception as e: # 捕获其他可能的错误 (如权限问题、字体问题等)
        print(f"❌ 生成或保存思维导图图像时发生意外错误: {e}")
        return None