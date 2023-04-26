import shutil
import time
import os

import streamlit as st
from PIL import Image
import base64

# импорт модуля распознования
import main

st.set_page_config(page_title="Рспознавание упражнений", layout="wide", page_icon="random")
import warnings
warnings.filterwarnings('ignore')

st.header('Сервис по распознаванию физических упражнений')

uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 300 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video

    dir_vid = 'users_video/'
    dir_img_with_points = 'users_imgs/'

    try:
        shutil.rmtree(dir_vid)
    except:
        pass

    try:
        shutil.rmtree(dir_img_with_points)
    except:
        pass

    if not os.path.isdir(dir_vid):
        os.mkdir(dir_vid)
    if not os.path.isdir(dir_img_with_points):
        os.mkdir(dir_img_with_points)


    vid = uploaded_video.name
    with open(f'{dir_vid}{vid}', mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    # Классификатор финально обучен на данном формате
    form_function = main.form_data1  # функция формирования numpy маасивов

    # Блок распознавания
    exec_time = time.time()
    res_table = main.classification(form_function,
                                    dir_vid,
                                    dir_img_with_points,
                                    pause_on_marked_up_photo=None)


    # Создание гивки и её отображение
    base_path_imgs = './users_imgs/'
    imgs_names = os.listdir(path=base_path_imgs)
    imgs_paths = [base_path_imgs + img_name for img_name in imgs_names]

    imgs_list = []
    for img_path in imgs_paths:
        img = Image.open(img_path)
        imgs_list.append(img)

    path_gif = 'users_imgs/fin_gif.gif'
    imgs_list[0].save(path_gif,
                      save_all=True, append_images=imgs_list[1:], optimize=False, duration=200, loop=0)

    """### gif from local file"""
    file_ = open(path_gif, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.table(data=res_table)