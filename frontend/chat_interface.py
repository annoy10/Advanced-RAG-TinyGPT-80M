from app_gradio import create_demo

def launch(workflow, **kwargs):
    return create_demo(workflow).launch(**kwargs)
