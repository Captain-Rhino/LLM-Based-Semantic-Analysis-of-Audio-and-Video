# --- test_keyframe_params.py ---
import math
import os
import json
import argparse
import shutil
import torch # 导入 torch 以便 KeyframeExtractor 初始化时找到设备
import cv2  # 导入 OpenCV 用于获取视频时长
from A_keyframe_extractor import KeyframeExtractor

def get_video_duration(video_path):
    """使用 OpenCV 获取视频总时长（秒）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频文件 {video_path} 来获取时长。")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 0 and frame_count_total > 0:
        return frame_count_total / fps
    else:
        print(f"警告：从 {video_path} 获取的 FPS ({fps}) 或总帧数 ({frame_count_total}) 无效。")
        return None

def test_parameters(video_path, asr_json_path, audio_path, interval_values, threshold_values, base_output_dir):
    """
    测试不同的 frame_interval 和 text_threshold 对关键帧数量和平均每帧代表时长的影响。
    """
    # === 新增：获取视频总时长 ===
    total_duration_seconds = get_video_duration(video_path)
    if total_duration_seconds is None:
        print("错误：无法获取视频总时长，测试中止。")
        return
    print(f"视频总时长: {total_duration_seconds:.2f} 秒")
    print("-" * 30)

    # results = {} # 改用列表存储更详细的结果
    results_list = [] # 用于存储结果字典列表

    # (文件检查和 ASR 数据加载保持不变)
    if not os.path.exists(video_path): print(f"错误：视频未找到: {video_path}"); return
    if not os.path.exists(asr_json_path): print(f"错误：ASR JSON 未找到: {asr_json_path}"); return
    if not os.path.exists(audio_path): print(f"错误：音频未找到: {audio_path}"); return
    try:
        with open(asr_json_path, 'r', encoding='utf-8') as f: asr_data = json.load(f)
        if not isinstance(asr_data, list): raise ValueError("ASR JSON 内容不是列表")
    except Exception as e: print(f"错误：加载 ASR JSON '{asr_json_path}' 失败: {e}"); return

    print(f"视频文件: {video_path}")
    print(f"ASR 文件: {asr_json_path}")
    print(f"音频文件: {audio_path}")
    print("-" * 30)
    print(f"待测试视觉间隔 (frame_interval): {interval_values}")
    print(f"待测试文本阈值 (text_threshold): {threshold_values}")
    print("-" * 30)

    extractor = KeyframeExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

    total_tests = len(interval_values) * len(threshold_values)
    current_test = 0
    for interval in interval_values:
        for threshold in threshold_values:
            current_test += 1
            print(f"\n[测试 {current_test}/{total_tests}] Interval={interval}, Threshold={threshold}")
            test_output_dir = os.path.join(base_output_dir, f"test_i{interval}_t{threshold}")
            os.makedirs(test_output_dir, exist_ok=True)
            print(f"  临时输出目录: {test_output_dir}")

            frame_count = 0
            avg_duration_per_frame = float('nan') # 初始化为 NaN

            try:
                keyframes = extractor.extract_keyframes(
                    video_path=video_path, output_dir=test_output_dir,
                    asr_data=asr_data, audio_path=audio_path,
                    frame_interval=interval, text_threshold=threshold
                )
                frame_count = len(keyframes)
                # === 新增：计算平均每帧时长 ===
                if frame_count > 0:
                    avg_duration_per_frame = total_duration_seconds / frame_count
                print(f"  提取关键帧数量: {frame_count}")
                print(f"  平均每帧代表时长: {avg_duration_per_frame:.2f} 秒")
                results_list.append({
                    "interval": interval, "threshold": threshold,
                    "count": frame_count, "avg_duration": avg_duration_per_frame
                })

            except Exception as e:
                print(f"  错误：提取失败: {e}")
                results_list.append({
                    "interval": interval, "threshold": threshold,
                    "count": f"Error", "avg_duration": float('nan')
                })
            finally:
                try:
                    # print(f"  清理临时目录: {test_output_dir}") # 可以取消注释以确认清理
                    shutil.rmtree(test_output_dir)
                except Exception as e_clean:
                    print(f"  警告：清理临时目录 {test_output_dir} 失败: {e_clean}")


    # --- 打印最终结果表格 ---
    print("\n" + "=" * 60)
    print("                          测试结果总结")
    print("=" * 60)
    print(f"{'视觉间隔':<12} {'文本阈值':<12} {'关键帧数量':<15} {'平均时长/帧(秒)':<18}")
    print("-" * 60)
    # 按 interval, threshold 排序打印
    results_list.sort(key=lambda x: (x["interval"], x["threshold"]))
    for result in results_list:
        interval = result["interval"]
        threshold = result["threshold"]
        count = result["count"]
        avg_duration = f"{result['avg_duration']:.2f}" if isinstance(result['avg_duration'], (int, float)) and not math.isnan(result['avg_duration']) else "N/A"
        print(f"{interval:<12} {threshold:<12} {count:<15} {avg_duration:<18}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试不同抽帧参数对关键帧数量和平均时长的影响。")
    parser.add_argument("--video", required=True, help="输入视频文件路径。")
    parser.add_argument("--asr", required=True, help="对应的 ASR JSON 文件路径。")
    parser.add_argument("--audio", required=True, help="对应的音频文件路径。")
    parser.add_argument("--intervals", nargs='+', type=int, default=[2], help="要测试的视觉间隔值列表 (通常固定为1个值)。") # 修改默认值，通常只测试一个 interval
    parser.add_argument("--thresholds", nargs='+', type=int, default=[30, 50, 80, 120], help="要测试的文本阈值列表。")
    parser.add_argument("--output", default="./keyframe_test_output", help="存放临时测试结果的基础目录。")

    args = parser.parse_args()
    if len(args.intervals) > 1:
        print("警告：通常建议固定视觉间隔 (intervals) 只测试一个值，以专注于文本阈值的影响。")

    os.makedirs(args.output, exist_ok=True)
    test_parameters(args.video, args.asr, args.audio, args.intervals, args.thresholds, args.output)