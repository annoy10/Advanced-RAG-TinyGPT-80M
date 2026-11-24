from main import build_runtime
from frontend.chat_interface import launch

if __name__ == "__main__":
    runtime = build_runtime()
    launch(runtime["workflow"], server_name="127.0.0.1", server_port=7860, inbrowser=True)
