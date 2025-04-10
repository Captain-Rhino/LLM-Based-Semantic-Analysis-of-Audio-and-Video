import gradio as gr

def greet(name):
    return f"你好，{name}！"

gr.Interface(fn=greet, inputs="text", outputs="text").launch()
