import gradio as gr
from config import analysis_history
from model_utils import load_models, predict_with_uncertainty
from image_utils import create_image_statistics_plot
from visualization import create_confidence_radar_chart

models = load_models()

with gr.Blocks(title="BREAST AI PRO") as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy")
            submit_btn = gr.Button("Analyze")
            output_text = gr.Markdown("Upload an image and click Analyze")
    def handle_submit(image):
        if image is None:
            return "Please upload an image"
        pred_idx, binary_conf, binary_uncertainty, subtype_name, subtype_conf, subtype_uncertainty, subtype_probs = predict_with_uncertainty(image)
        analysis_history.append({'diagnosis': subtype_name, 'confidence': binary_conf})
        return f"Diagnosis: {subtype_name}, Conf: {binary_conf:.2f}% ± {binary_uncertainty:.2f}%"
    submit_btn.click(fn=handle_submit, inputs=[image_input], outputs=[output_text])
