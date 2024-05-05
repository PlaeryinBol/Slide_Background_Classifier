import albumentations as A
import gradio as gr
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

import config
import models

model = models.ClassificationModel(config.MODEL_NAME, apply_softmax=True).to(config.DEVICE)
model.load_state_dict(torch.load(config.EXP_NAME + '_best.pth'))


def preprocess_image(image):
    transform = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE, p=1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)
    ])
    image = transform(image=image)["image"].unsqueeze(0).to(config.DEVICE)
    return image


def predict_image(image: Image.Image, apply_smoothing=True):
    image = preprocess_image(image)

    with torch.no_grad():
        model.eval()
        test_output = model(image)
        predict = torch.argmax(test_output, axis=1)
        if apply_smoothing:
            confident_predict = test_output[:, 1] > config.SOFTMAX_THRESHOLD
            predict = torch.where(confident_predict, torch.tensor(1), torch.tensor(0)).tolist()

    cls = 'good' if predict[0] else 'bad'
    return cls


if __name__ == '__main__':
    iface = gr.Interface(fn=predict_image,
                         inputs=gr.inputs.Image(),
                         outputs="text",
                         live=True,
                         allow_flagging='never',
                         title="Photobackground classifier",
                         description="Load image to classify it as good slide background or not.",
                         examples=["./examples/bad_1.jpg", "./examples/good_1.jpg",
                                   "./examples/bad_2.jpg", "./examples/good_2.jpg"])
    iface.launch()
