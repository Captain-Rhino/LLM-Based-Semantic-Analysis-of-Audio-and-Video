import subprocess

# 定义你想要执行的命令
command = ["python", "G:/videochat/SenseVoice/speech_server_IN_SENSEVOICE.py"]

# 使用 subprocess 执行命令
try:
    subprocess.run(command, check=True)
    print("Server started successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while starting the server: {e}")
