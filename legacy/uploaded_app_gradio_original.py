
import gradio as gr

from model import chat


# ============================================================
# CHAT FUNCTION
# ============================================================

def chatbot_response(message, history):

    if not message.strip():
        return "Please ask a sports question."

    try:
        answer = chat(message)
        return answer

    except Exception as e:
        return f"Error: {e}"


# ============================================================
# GRADIO UI
# ============================================================

demo = gr.ChatInterface(
    fn=chatbot_response,
    title="⚽ TinyGPT Sports QA Chatbot",
    description="Ask a sports question to TinyGPT.",
    textbox=gr.Textbox(
        placeholder="Ask a sports question...",
        container=True
    )
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )

