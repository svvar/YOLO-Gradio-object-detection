import cv2
import gradio as gr
import validators
from cap_from_youtube import cap_from_youtube

import video_detection
from video_detection import video_detection


def preprocess_input(input):
    # if not input:
    #     cap = cv2.VideoCapture(0)
    #     yield from video_detection(cap)
    if validators.url(input):
        if 'youtu' in input:
            cap = cap_from_youtube(input, resolution='720p')
            yield from video_detection(cap)
        else:
            print("Invalid URL")
    else:
        cap = cv2.VideoCapture(input)
        yield from video_detection(cap)

# gradio interface
input_video = gr.Video(label="Input Video")
input_url = gr.Textbox(label="Input URL", placeholder="Enter URL")
output_frames_1 = gr.Image(label="Output Frames")
output_video_file_1 = gr.Video(label="Output video")
output_frames_2 = gr.Image(label="Output Frames")
output_video_file_2 = gr.Video(label="Output video")
# sample_video=r'sample/car.mp4'

file_tab = gr.Interface(
    fn=preprocess_input,
    inputs=[input_video],
    outputs=[output_frames_1, output_video_file_1],
    title=f"Завантажте файл для розпізнавання",
    allow_flagging="never",
    examples=[["car.mp4"]],
)

url_tab = gr.Interface(
    fn=preprocess_input,
    inputs=[input_url],
    outputs=[output_frames_2, output_video_file_2],
    title=f"Введіть URL Youtube відео для розпізнавання",
    allow_flagging="never",
    examples=[["car.mp4"]],
)

app = gr.TabbedInterface([file_tab, url_tab], ["Завантажити файл", "Ввести URL"])

app.launch()
