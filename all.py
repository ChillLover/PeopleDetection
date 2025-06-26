import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import uuid
import os
import atexit
import gradio as gr
from moviepy.editor import ImageSequenceClip


model = YOLO("yolo12n.pt")
names = model.names
files = []


def clean():
    """
    Функция очистки всех предсказаний после окончания работы программы.
    """
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            continue


atexit.register(clean)


def prediction(video, conf_level):
    """
    Функция обрабатывает видео, выполняет детекцию людей и их отрисовку на видео, сохраняя результаты.
    Args: 
        - video: Видео, на котором требуется распознать людей.
        - conf_level: Значение параметра уверенности модели.
    Returns: 
        - output_name: Видео с распознанными людьми.
        - output_name: Видео с распознанными людьми для корректной работы с Gradio.
    """
    cap = cv2.VideoCapture(video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    output_name = f"preds_{uuid.uuid4()}.mp4"
    files.append(output_name)
    #writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))

    frames = []

    while True:
        ret, img = cap.read()
        if not ret:
            break

        annotator = Annotator(img)
        results = model.predict(img, classes=[0], conf=conf_level)
        boxes = results[0].boxes.xyxy.cpu()
        classes = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for indx, (box, cls) in enumerate(zip(boxes, classes)):
            label = f"{names[int(cls)]}: {np.round(confs[indx], 2)}"
            annotator.box_label(box, label=label, color=(0, 0, 255),) #label=names[int(cls)]

        current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(current_frame)

    cap.release()

    if len(frames) > 0:
        clip = ImageSequenceClip(frames, fps)
        clip.write_videofile(output_name, codec="libx264", threads=4, audio=False)

    return output_name, output_name


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_file = gr.Video(label="Загрузить видео")
            conf_level = gr.Slider(
                label="Уверенность модели",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.45,
            )
            process_button = gr.Button("Обработать видео")

        with gr.Column():
            output = gr.Video(label="Полученные предсказания")
            download_button = gr.DownloadButton(label="Скачать видео")

    process_button.click(
        fn=prediction,
        inputs=[input_file, conf_level],
        outputs=[output, download_button],
    )

demo.launch(server_name="0.0.0.0")
