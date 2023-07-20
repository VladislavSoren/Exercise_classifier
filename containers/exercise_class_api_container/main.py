import shutil
import time
import os
from pathlib import Path

import uvicorn
from PIL import Image
import base64

from fastapi import FastAPI
from pydantic import BaseModel

# импорт модуля распознования
import classification_module

import warnings

warnings.filterwarnings('ignore')

app = FastAPI()


# Request processing to the service root 
@app.get("/")
def index():
    return {
        "message: Index!"
    }


class PredictRequest(BaseModel):
    user: str
    video: str
    video_name: str


def media_bytes_to_str(im_path):
    with open(im_path, mode='rb') as file:
        media_bytes = file.read()
    media_str = base64.encodebytes(media_bytes).decode('utf-8')
    return media_str


@app.post("/video")
def get_video(json_input: PredictRequest):
    #  deserialization
    video_bytes = base64.b64decode(json_input.video)
    video_name = json_input.video_name

    # recreate dir_vid and dir_img_with_points dirs
    dir_vid = 'users_video/'
    dir_img_with_points = 'users_imgs/'
    if os.path.isdir(dir_vid): shutil.rmtree(dir_vid)
    if os.path.isdir(dir_img_with_points): shutil.rmtree(dir_img_with_points)
    if not os.path.isdir(dir_vid): os.mkdir(dir_vid)
    if not os.path.isdir(dir_img_with_points): os.mkdir(dir_img_with_points)

    # saving video
    with open(f'{dir_vid}{video_name}', mode='wb') as f:
        f.write(video_bytes)  # save video to disk

    # specify the type of the formation function numpy dataframes
    form_function = classification_module.form_data1

    # receiving prediction
    res_table = classification_module.classification(form_function,
                                                     dir_vid,
                                                     dir_img_with_points,
                                                     pause_on_marked_up_photo=None)

    # parsing prediction
    pred_type = res_table['type of exercise'].values[0]
    pred_count = res_table['number of repetitions'].values[0]

    # gif creation
    base_path_imgs = './users_imgs/'
    imgs_names = os.listdir(path=base_path_imgs)
    imgs_paths = [base_path_imgs + img_name for img_name in imgs_names]

    imgs_list = []
    for img_path in imgs_paths:
        img = Image.open(img_path)
        imgs_list.append(img)

    # save tagged images as a gif
    path_gif = f'users_imgs/{video_name}_gif.gif'
    imgs_list[0].save(path_gif,
                      save_all=True, append_images=imgs_list[1:], optimize=False, duration=200, loop=0)

    # serialization response
    json_out = {}
    json_out['pred_type'] = pred_type
    json_out['pred_count'] = pred_count
    json_out['tagged_gif'] = media_bytes_to_str(path_gif)

    return json_out


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=9988,
        # reload=True,
    )
