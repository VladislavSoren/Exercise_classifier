#############################################
# подгружаем необходимые библиотеки и модули
#############################################
from matplotlib import pyplot as plt
from matplotlib import image
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import cv2
from PIL import Image
import numpy as np
import os
import keras as K
import subprocess
import  time
import  tkinter as tk
from tkinter import filedialog
import os
import shutil
import  pandas as pd

# инструменты нормализации
from sklearn import preprocessing

#############################
# Загружаем предопученные НС
#############################

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)  # для определение координат бокса
pose_net = model_zoo.get_model('simple_pose_resnet152_v1d', pretrained=True)  # для определения координат основных узлов

#########################################################################
# подстраиваем yolo3 на поиск человека (используем соответствующие веса)
#########################################################################
detector.reset_class(["person"], reuse_weights=['person'])

########################################
# загружаем предобученный классификатор
########################################
# получаем путь к директории с классификатором
path_classificator = filedialog.askopenfilename(title='Выберите классификатор')
exercise_recognizer = K.models.load_model(path_classificator)

"""# Необходимые функции"""


######################################
# функция предикта одного изображения
######################################
def predict(path):
    # path - путь к файлу
    x, img = data.transforms.presets.ssd.load_test(path, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    print('Координаты бокса: ', upscale_bbox)
    print('Размер бокса: ', len(upscale_bbox))

    print('Суставные координаты: ', pred_coords)
    print('Размер координат: ', pred_coords.shape)

    ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                                  class_IDs, bounding_boxs, scores,
                                  box_thresh=0.5, keypoint_thresh=0.2)

    ax  # наверное отрисовка
    plt.draw()
    plt.gcf().canvas.flush_events()

    # Задержка перед следующим обновлением
    time.sleep(0.01)

    # закрываем картинку
    plt.close()

    # возвращаем координаты бокса и узлов
    return upscale_bbox, pred_coords


################################################
# функция получения длины обрабатываемого видео
################################################
'''
def get_length(filename):
    print(filename)
    print(filename)
    print(filename)
    print(filename)
    print(filename)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)
'''

########################
# функция нарезки видео
########################
def save_frames(dir_video, dir_img, min_value):
    list_video_names = os.listdir(dir_video)

    video_length = []
    all_FPS = []
    number_of_cadrs = []
    bad_video = []

    for cur_video in list_video_names:
        cur_video_path = dir_video + cur_video
 #       cur_length = get_length(cur_video_path)
 #       print('Длина видео', cur_length)

        video_capture = cv2.VideoCapture(cur_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print('FPS - ', fps)
        video_capture.set(cv2.CAP_PROP_FPS, fps)

        counter = 0

        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()
            cur_video = cur_video.replace(".", "")                                          # .replace(".", "") исключает точки из названия видео
            if frame_is_read:
                print(type(frame))
                cv2.imwrite(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg', frame)
                print(cv2.imwrite(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg', frame))
#                frame = frame.astype(np.uint8)
#                image.imsave(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg', frame)
                counter += 1
            else:
                print("Could not read the frame.")
                break
        print('Количество кадров: ', counter)
        print(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg')
#        video_length.append(cur_length)
        all_FPS.append(fps)
        number_of_cadrs.append(counter)

        if counter < min_value:
            bad_video.append(cur_video)

    list_img = os.listdir(dir_img)

#    print(min(video_length), max(video_length), video_length)
    print(min(all_FPS), max(all_FPS), all_FPS)
    print(min(number_of_cadrs), max(number_of_cadrs), number_of_cadrs)
    print(bad_video)

    return list_video_names


############################################################
# функция выделения признаков из нужных кадров ОДНОГО видео
############################################################

def show_pred(dir_source, list_video_names, function):
    data_all = np.array([])  # массив данных по всем видео

    for video_name in list_video_names:
        list_bbox = []

        list_coords = []

        # !!! Включить интерактивный режим для анимации
        plt.ion()

        video_name = video_name.replace(".", "")  # .replace(".", "") исключает точки из названия видео

        # проходимся по всем нужным кадрам текущего видео
        for counter in range(0, 100, 10):
            #      img = plt.imread( f'{dir_source}{video_name}_cadr_{str(counter)}.jpg')
            img_path = f'{dir_source}{video_name}_cadr_{str(counter)}.jpg'

            count = 0
            for try_step in range(10):
                try:
                    upscale_bbox, pred_coords = predict(f'{dir_source}{video_name}_cadr_{str(counter + count)}.jpg')
                except:
                    count += 1
                    continue
                else:
                    break
            count = 0

            list_bbox.append(upscale_bbox)
            list_coords.append(pred_coords)

        plt.ioff()
#        plt.show()

        # Получаем сформированные данны текущего кадра
        data_video = function(list_bbox, list_coords)

        data_all = np.concatenate((data_all, data_video), axis=0)

    return data_all, len(list_video_names)


################################################################
# функция №1 преобразования признаков текущего видео в Data данные
################################################################

def form_data1(list_bbox, list_coords):
    data = np.array([])

    for i in range(len(list_bbox)):
        bbox_np1 = np.array(list_bbox[i][0])
        bbox_np1 = bbox_np1.reshape(2, 2)

        coords1 = list_coords[i][0]
        coords_np1 = coords1.asnumpy()

        # concatination
        concat1 = np.concatenate((bbox_np1, coords_np1), axis=0)
        concat1 = concat1.reshape(38)

        data = np.concatenate((data, concat1), axis=0)

    return data


################################################################
# функция №2 преобразования признаков текущего видео в Data данные
################################################################
def form_data2(list_bbox, list_coords):
    data = np.array([])

    for i in range(1, len(list_bbox)):
        bbox_np1 = np.array(list_bbox[i - 1][0])
        bbox_np2 = np.array(list_bbox[i][0])
        dif_bbox = bbox_np2 - bbox_np1
        dif_bbox1 = dif_bbox.reshape(2, 2)

        coords1 = list_coords[i - 1][0]
        coords2 = list_coords[i][0]
        coords_np1 = coords1.asnumpy()
        coords_np2 = coords2.asnumpy()
        dif_coors1 = coords_np2 - coords_np1

        # concatination
        concat1 = np.concatenate((dif_bbox1, dif_coors1), axis=0)
        concat1 = concat1.reshape(38)

        data = np.concatenate((data, concat1), axis=0)

    return data


####################################################
# получения предикта классификатора из данных numpy
####################################################
def control_predict(data_all, list_len):
    # нормализация данных
    data_norm = preprocessing.minmax_scale(data_all.T).T

    # изменение размерности
    data_all_reshape = data_norm.reshape(list_len, 10, 38)
    data_all_reshape.shape

    # добавление размерности для Conv слоёв
    x_test = data_all_reshape.reshape(list_len, 10, 38, 1)
    print(x_test.shape)

    # вывод определенного класса
    names = ['приседания', 'подтягивания', 'отжимания']
    max_indent = 15

    prediction = exercise_recognizer.predict(x_test)  # получаем весь предикт

    df = pd.DataFrame(columns=['predict'])

    Class_img_path = filedialog.askdirectory(title='Укажите путь к картинкам классов')

    for i in range(list_len):
        cluss_number = np.argmax(prediction[i])
        pred = names[cluss_number]

        print(f'Распознанный класс:{pred:<{max_indent}}')

        img = Image.open(f'{Class_img_path}/{cluss_number}.jpg')

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()
        ax.imshow(img)
        ax.set(title=f'Видео № {i+1}')
        plt.axis('off')

        plt.show()

        df.loc[f'Video_{i + 1}'] = pd.Series(
            {'predict': pred})

    window = tk.Tk()
    window.geometry('500x500')
    window.title("Таблица", )

    lbl = tk.Label(window, text=df, font=("Arial Bold", 25))
    lbl.grid(column=0, row=20)

    window.mainloop()
####################################
# обобщение всех глобальных функций
####################################
def classification(form_data1, min_cadrs_value = 95):
    # получаем путь к директории с видео
    dir_vid = filedialog.askdirectory()
    dir_vid = f'{dir_vid}/'

    # на основе него задаем путь для сохранения картинок
    dir_img = 'dir_img/'                                 # не забывать про /
#    dir_img = f'{dir_vid[:-8]}{dir_img}/'

    print(dir_img)

    # повторный запуск mkdir с тем же именем вызывает FileExistsError,
    # вместо этого запустите:
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)

    # нарезка видео и сохранение кадров
    list_video_names = save_frames(dir_vid, dir_img, min_cadrs_value)

    # Получение из кадров видео nampy массива
    data_all, list_len = show_pred(dir_img, list_video_names, form_data1)

    # получение предсказания
    control_predict(data_all, list_len)

    # в конце удаляем папку с нарезанными кадрами (vid_img)
    shutil.rmtree(dir_img)