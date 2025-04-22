# from A_clip_finetune import train_adaptor
#
# model = train_adaptor(
#     data_path=r"G:\videochat\my_design\CNCLIP_keyframes_test_video\clip_features.pth",
#     output_dir="adaptor_results",
#     epochs=100,
#     batch_size=4  # 你只有11条数据，调小一点
# )
#
import face_recognition
import cv2
import os

def detect_and_save_faces_accurate(image_path, output_dir="faces"):
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ 无法读取图片：{image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 使用 face_recognition 检测人脸
    face_locations = face_recognition.face_locations(image_rgb, model='cnn')  # 选择 cnn 模型更准确

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    for idx, (top, right, bottom, left) in enumerate(face_locations):
        face_crop = image_bgr[top:bottom, left:right]
        save_path = os.path.join(output_dir, f"{base_filename}_face{idx}.jpg")
        cv2.imwrite(save_path, face_crop)
        print(f"✅ 已保存人脸: {save_path}")

    if len(face_locations) == 0:
        print("⚠️ 未检测到人脸")

# === 示例调用 ===
input_image = r"G:\videochat\my_design\CNCLIP_keyframes_test_movie\comp_kf_01790.jpg"
output_dir = r"G:\videochat\my_design\CNCLIP_keyframes_test_2\faces"

detect_and_save_faces_accurate(input_image, output_dir)

