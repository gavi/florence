import os
import requests
from PIL import Image, ImageDraw
from unittest.mock import patch
import gradio as gr
import ast
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

def draw_boxes(image, boxes, box_type='bbox', labels=None):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        if box_type == 'quad':
            draw.polygon(box, outline="red", width=2)
        elif box_type == 'bbox':
            draw.rectangle(box, outline="red", width=2)
        
        if labels and i < len(labels):
            draw.text((box[0], box[1] - 10), labels[i], fill="red")
    
    return image

def run_example(image, task, additional_text=""):
    if image is None:
        return "Please upload an image.", None

    prompt = f"<{task}>"
    if task == "CAPTION_TO_PHRASE_GROUNDING" and additional_text:
        inputs = processor(text=prompt, images=image, return_tensors="pt", text_input=additional_text)
    else:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    
    result_text = str(parsed_answer)
    result_image = image.copy()
    
    try:
        result_dict = ast.literal_eval(result_text)
        task_key = f"<{task}>"
        if task_key in result_dict:
            if 'quad_boxes' in result_dict[task_key]:
                result_image = draw_boxes(result_image, result_dict[task_key]['quad_boxes'], 'quad')
            elif 'bboxes' in result_dict[task_key]:
                result_image = draw_boxes(result_image, result_dict[task_key]['bboxes'], 'bbox', result_dict[task_key].get('labels'))
    except:
        print(f"Failed to draw bounding boxes for task: {task}")
    
    return result_text, result_image

def update_additional_text_visibility(task):
    return gr.update(visible=(task == "CAPTION_TO_PHRASE_GROUNDING"))

# Define the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Florence-2 Image Analysis")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        with gr.Column():
            task_dropdown = gr.Dropdown(
                choices=[
                    "CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION",
                    "CAPTION_TO_PHRASE_GROUNDING", "OD", "DENSE_REGION_CAPTION",
                    "REGION_PROPOSAL", "OCR", "OCR_WITH_REGION"
                ],
                label="Select Task",
                value="CAPTION"
            )
            additional_text = gr.Textbox(
                label="Additional Text (for Caption to Phrase Grounding)",
                placeholder="Enter caption here",
                visible=False
            )
            submit_button = gr.Button("Analyze Image")
    with gr.Row():
        text_output = gr.Textbox(label="Result")
        image_output = gr.Image(label="Processed Image")

    task_dropdown.change(fn=update_additional_text_visibility, inputs=task_dropdown, outputs=additional_text)
    submit_button.click(
        fn=run_example,
        inputs=[image_input, task_dropdown, additional_text],
        outputs=[text_output, image_output]
    )

# Launch the interface
iface.launch()