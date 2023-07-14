import shutil
import time
import os
from pathlib import Path

import streamlit as st
from PIL import Image
import base64

# импорт модуля распознования
import classification_module

import warnings

warnings.filterwarnings('ignore')


@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)

    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


# Функция получения html таблицы предсказаний
def get_html_pred_table(pred_type, pred_count):
    html_code = f"""
    <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; font-size:150%;">
    <thead>
    <tr style="background-color: #fcf403;">
        <th style="padding: 0 1em 0 0.5em; text-align: center;">тип упражнения</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: center;">количество</th>
    </tr>
    </thead>
    <tbody>
        <tr style="background-color: #fc9803; border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: center;">
                {pred_type}
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: center;">
                {pred_count}
            </td>
        </tr>
    </tbody>
    """
    return html_code


# Конфигурирование страницы
im = Image.open(Path.cwd() / 'APP_icon' / 'Иконка.png')
st.set_page_config(page_title="Рспознавание_упражнений", layout="wide", page_icon=im)

# Устанавливаем фон
set_png_as_page_bg(Path.cwd() / 'APP_bg' / 'Bg.jpg')

st.header('Сервис по распознаванию физических упражнений')

url = 'https://t.me/VladislavSoren'
full_ref = f'<a href="{url}" style="color: #0d0aab">by FriendlyDev</a>'
st.markdown(f"<h2 style='font-size: 20px; text-align: right; color: black;'>{full_ref}</h2>", unsafe_allow_html=True)

uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 300  # display every 300 frames

if uploaded_video is not None:  # run only when user uploads video

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

    if not os.path.isdir(dir_vid): os.mkdir(dir_vid)
    if not os.path.isdir(dir_img_with_points): os.mkdir(dir_img_with_points)

    vid = uploaded_video.name
    with open(f'{dir_vid}{vid}', mode='wb') as f:
        f.write(uploaded_video.read())  # save video to disk

    # Классификатор финально обучен на данном формате
    form_function = classification_module.form_data1  # функция формирования numpy маасивов

    # Блок распознавания
    exec_time = time.time()
    res_table = classification_module.classification(form_function,
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

    file_ = open(path_gif, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    pred_type = res_table['type of exercise'].values[0]
    pred_count = res_table['number of repetitions'].values[0]
    html_code = get_html_pred_table(pred_type, pred_count)

    # Отображаем таблицу предсказания и гифку
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

else:
    st.header('preview')

    html_code = get_html_pred_table('подтягивания', '3')

    st.markdown(html_code, unsafe_allow_html=True)

    path_gif = 'preview/fin_gif.gif'

    file_ = open(path_gif, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
