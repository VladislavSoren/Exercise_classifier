#############################################
# Подгружаем необходимые библиотеки и модули.
#############################################

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import keras as K
from sklearn import preprocessing
import cv2

from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

import time
import os
import shutil
import pandas as pd

#############################
# Загружаем предопученные НС.
#############################

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)  # yolo3 определяет координаты бокса на кадре
pose_net = model_zoo.get_model('simple_pose_resnet152_v1d',
                               pretrained=True)  # simple_pose определяет координаты основных узлов

#########################################################################
# Подстраиваем yolo3 на поиск человека (используем соответствующие веса).
#########################################################################

detector.reset_class(["person"], reuse_weights=['person'])

########################################
# Загружаем предобученный классификатор.
########################################
path_classificator = 'exercise_recognizer_best.h5'
exercise_recognizer = K.models.load_model(path_classificator)

"""_______________________________________________Функции проекта____________________________________________"""


######################################
# функция предикта одного изображения.
######################################
def predict(path,
            dir_img_with_points,
            counter,
            pause_on_marked_up_photo=None):
    '''
    Значение функции:
        Получить координаты bounding_boxs и upscale_bbox с одного кадра

    Входные данные:
        path - ссылка на конкретный кадр 

    Входные данные:
        upscale_bbox - координаты bounding_boxs
        pred_coords  - координаты upscale_bbox 
    '''

    x, img = data.transforms.presets.ssd.load_test(path, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    # Данные по кадру
    # print('Координаты бокса: ', upscale_bbox)
    # print('Размер бокса: ', len(upscale_bbox))
    # print('Суставные координаты: ', pred_coords)
    # print('Размер координат: ', pred_coords.shape)

    # Если не была введена пауза для отображения размеченных кадров,
    if pause_on_marked_up_photo != None:
        ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                                      class_IDs, bounding_boxs, scores,
                                      box_thresh=0.5, keypoint_thresh=0.2)

    # if (counter > 0) and ((counter < 40)):
    if (counter > 0) and (counter < 1000):
        img = utils.viz.cv_plot_keypoints(img, pred_coords, confidence,
                                          class_IDs, bounding_boxs, scores,
                                          box_thresh=0.5, keypoint_thresh=0.2)

        cv2.imwrite(f'{dir_img_with_points}img_{counter}.jpg', img)

        # ax
        # plt.draw()
        # plt.gcf().canvas.flush_events()

        # Задержка перед следующим обновлением
        if pause_on_marked_up_photo != None:
            time.sleep(pause_on_marked_up_photo)

        # закрываем картинку
        plt.close()

    # возвращаем координаты бокса и узлов
    return upscale_bbox, pred_coords


########################
# Функция нарезки видео.
########################

def save_frames(dir_video, dir_img, min_value):
    '''
    Значение функции:
        Нарезка всех видно на кадры и их сохранение 

    Входные данные:
        dir_video - ссылка директорию с видео 
        dir_img   - ссылка на директорию с кадрами 
        min_value - минимальное кол-во кадров в видео 

    Входные данные:
        list_video_names - список имен видео
        number_of_cadrs  - список количества кадров всех видео [108, 99, 105...]
    '''

    list_video_names = os.listdir(dir_video)
    all_FPS = []
    number_of_cadrs = []
    bad_video = []

    for cur_video in list_video_names:

        cur_video_path = dir_video + cur_video
        video_capture = cv2.VideoCapture(cur_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print('FPS - ', fps)
        video_capture.set(cv2.CAP_PROP_FPS, fps)

        counter = 0

        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()
            cur_video = cur_video.replace(".", "")
            if frame_is_read:
                cv2.imwrite(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg', frame)
                counter += 1
                if counter % 20 == 0:
                    print(counter)
            else:
                print("Could not read the frame.")
                break

        print('Количество кадров: ', counter)
        print(f'{dir_img}{cur_video}_cadr_{str(counter)}.jpg')
        all_FPS.append(fps)
        number_of_cadrs.append(counter)

        if counter < min_value:
            bad_video.append(cur_video)

    list_img = os.listdir(dir_img)

    print(min(all_FPS), max(all_FPS), all_FPS)
    print(min(number_of_cadrs), max(number_of_cadrs), number_of_cadrs)
    print(bad_video)

    return list_video_names, number_of_cadrs


#############################################################
# Функция получения DataFrames для классификатора и счетчика.
#############################################################

def show_pred(dir_source,
              dir_img_with_points,
              list_video_names,
              number_of_cadrs,
              need_count,
              function,
              pause_on_marked_up_photo):
    '''
    Значение функции:
        Выделения признаков из кадров ВСЕХ видео для классификатора и счетчика

    Входные данные:
        dir_source       - ссылка на директорию с кадрами 
        list_video_names - список имен видео
        number_of_cadrs  - список количества кадров всех видео [108, 99, 105...]
        need_count       - флаг выбора режима (если need_count==1, то помимо классификации 
                                             так же будет вестись подсчет количества повторений)
        function         - функция преобразования признаков текущего видео в Data данные 
        
    Входные данные:
        data_class            - сформированные данные для классификатора
        data_NoR              - сформированные данные для счетчика 
        len(list_video_names) - общее количество видеофайлов
    '''

    data_class = np.array([])  # массив данных для классификатора
    data_NoR = []  # массив количества повторов на всех видео

    # Цикл формирования данных по ВСЕМ видео 
    for video_name in list_video_names:

        # листы для классификации
        list_bbox = []
        list_coords = []

        # листы для подсчета 
        list_bbox_count = []
        list_coords_count = []

        # Включение интерактивного режима для анимации
        plt.ion()

        video_name = video_name.replace(".", "")

        if need_count:  # Если выбран режим подсчтета количества упражнений

            print(number_of_cadrs, type(number_of_cadrs))

            # Цикл формирования данных с кадров ОДНОГО видео
            for counter in range(0, max(number_of_cadrs), 10):

                img_path = f'{dir_source}{video_name}_cadr_{str(counter)}.jpg'

                count = 0
                for try_step in range(10):
                    try:
                        upscale_bbox, pred_coords = predict(f'{dir_source}{video_name}_cadr_{str(counter + count)}.jpg',
                                                            dir_img_with_points,
                                                            counter,
                                                            pause_on_marked_up_photo=pause_on_marked_up_photo)
                    except:
                        count += 1
                        continue
                    else:
                        break
                count = 0

                # получение координат для классификации
                list_bbox_count.append(upscale_bbox)
                list_coords_count.append(pred_coords)

            # получение координат для подсчета
            list_bbox = list_bbox_count[0:10]
            list_coords = list_coords_count[0:10]

            # print('list_bbox_count:', list_bbox_count)
            # print('list_coords_count', list_coords_count)
            # print('list_bbox', list_bbox)
            # print('list_coords', list_coords)

        else:  # Если НЕ выбран режим подсчтета количества упражнений

            for counter in range(0, 100, 10):

                img_path = f'{dir_source}{video_name}_cadr_{str(counter)}.jpg'

                count = 0
                for try_step in range(10):
                    try:
                        upscale_bbox, pred_coords = predict(f'{dir_source}{video_name}_cadr_{str(counter + count)}.jpg',
                                                            dir_img_with_points,
                                                            counter,
                                                            pause_on_marked_up_photo=pause_on_marked_up_photo)
                        # размерность pred_coords - (1, 17, 2)
                    except:
                        count += 1
                        continue
                    else:
                        break
                count = 0

                # получение координат для классификации
                list_bbox.append(upscale_bbox)
                list_coords.append(pred_coords)

        plt.ioff()

        # ..................................Данные для классификатора.........................
        # Получаем сформированные данны текущего видео по данным для классификации
        data_video = function(list_bbox, list_coords)
        print(data_video)

        # Конкатинируем все видео и получаем общий массив с данными
        data_class = np.concatenate((data_class, data_video), axis=0)  # ()
        print(data_class)

    # ..................................Данные для счетчика.........................
    # # Получаем сформированные данны текущего видео по данным для классификации
    # number_of_repet = function_count(list_coords_count)
    #
    # # Конкатинируем все видео и получаем общий массив с данными
    # data_NoR.append(number_of_repet)
    # print('data_NoR:', data_NoR)

    return data_class, list_coords_count, len(list_video_names)

    ###################################################################
    # Функция подсчета количества повторений упражнения на ОДНОМ видео.
    ###################################################################

    '''
    Значение функции:
        Подсчет количества повторений на одном видео

    Входные данные:
        list_coords_count - массив координат узлов на теле человека

    Выходные данные:
        count - количество повторений упражнения на текущем видео
    '''


def function_count(list_coords_count, pred):
    list_y = []
    for cadr in list_coords_count:  # cadr.shape = (1, 17, 2)
        list_y.append(cadr[0][11][1])  # берём координаты узла таза на наиболее вероятном человеке в кадре

    # считаем кол-во повторений
    if pred == 'подтягивания':
        check = True
        count = 0

        for i in range(1, len(list_y)):
            filtr = (abs(list_y[i] - list_y[i - 1]) > 15)

            if (list_y[i] - list_y[i - 1] < 0) and filtr:
                check = True

            if check:
                if (list_y[i] - list_y[i - 1] >= 0) and filtr:
                    check = False
                    count += 1
    else:
        check = True
        count = 0

        for i in range(1, len(list_y)):
            filtr = (abs(list_y[i] - list_y[i - 1]) > 15)

            if (list_y[i] - list_y[i - 1] > 0) and filtr:
                check = True

            if check:
                if list_y[i] - list_y[i - 1] <= 0 and filtr:
                    check = False
                    count += 1

    print(count)
    return count


###################################################################
# Функция №1 преобразования признаков текущего видео в Data данные.
###################################################################

def form_data1(list_bbox, list_coords):
    '''
    Значение функции:
        слить данные после предобученных нейронок воедино

    Входные данные:
        list_bbox - лист типа [[376.4499092102051, 34.83863830566406, 564.3607902526855, 501.78428649902344]]
        list_coords - np массив размерностью (1, 17, 2), где 1 - кол-во найденных объектов

    Входные данные:
        data - np массив размерность (380, )
    '''

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

    ###################################################################
    # Функция №2 преобразования признаков текущего видео в Data данные.
    ###################################################################

    '''
    Значение функции:
        Слить данные после предобученных нейронок воедино. В отличае от функции №1
        берутся не фактические значения координат, а их разности.

    Входные данные:
        list_bbox - лист типа [[376.4499092102051, 34.83863830566406, 564.3607902526855, 501.78428649902344]]
        list_coords - np массив размерностью (1, 17, 2), где 1 - кол-во найденных объектов

    Выходные данные:
        data - np массив размерность (380, )
    '''


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


#############################################################
# Получения предикта классификатора из Data (numpy массивов).
#############################################################

def control_predict(data_all, list_coords_count, list_len):
    '''
    Значение функции:
        Предсказывание типа упражнения (классификация) 

    Входные данные:
        data_class            - сформированные данные для классификатора
        data_NoR              - сформированные данные для счетчика 
        list_len              - общее количество видеофайлов

    '''

    # нормализация данных
    data_norm = preprocessing.minmax_scale(data_all.T).T

    # ..........................подготовка данных для классификатора........................
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

    df = pd.DataFrame(columns=['type of exercise', 'number of repetitions'])

    # Class_img_path = filedialog.askdirectory(title='Укажите путь к картинкам классов')
    Class_img_path = 'Class_img'

    for i in range(list_len):
        cluss_number = np.argmax(prediction[i])
        pred = names[cluss_number]

        print(f'Распознанный класс:{pred:<{max_indent}}')

        # img = Image.open(f'{Class_img_path}/{cluss_number}.jpg')

        # fig = plt.figure(figsize=(6, 4))
        # ax = fig.add_subplot()
        # ax.imshow(img)
        # ax.set(title=f'Видео № {i+1}')
        # plt.axis('off')

        # plt.show()

        # Подсчёт кол-во выполненного упражнения
        # Получаем сформированные данны текущего видео по данным для классификации
        number_of_repet = function_count(list_coords_count, pred)

        df.loc[f'Video_{i + 1}'] = pd.Series(
            {'type of exercise': pred, 'number of repetitions': number_of_repet})

    # # Конкатинируем все видео и получаем общий массив с данными
    # data_NoR.append(number_of_repet)
    # print('data_NoR:', data_NoR)

    return df


####################################
# Слияние всех глобальных функций.
####################################

def classification(form_data1,
                   dir_vid,
                   dir_img_with_points,
                   min_cadrs_value=95, pause_on_marked_up_photo=None):
    '''
    Значение функции:
        Слияние всех глобальных функций (классификация и подсчет)

    Входные данные:
        form_data1      - преобразования признаков (предсказанных координат) текущего видео в Data данные
        min_cadrs_value - минимальный порог количество кадров в видео

    '''

    # Default
    need_count = '1'

    # на основе него задаем путь для временного хранения кадров видео
    dir_img = 'dir_img/'  # не забывать про "/"

    print(dir_img)

    # Если dir_img нет, то создаём её
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)

    # нарезка видео и сохранение кадров
    list_video_names, number_of_cadrs = save_frames(dir_vid,
                                                    dir_img,
                                                    min_cadrs_value)
    # Выбор режима
    # need_count = str(input('Введите 1, чтобы активировать режим "Классификация + счет"\n'))
    print(need_count)
    if need_count == '1':
        need_count = True
    else:
        need_count = False

    # Получение из кадров видео nampy массива
    data_all, list_coords_count, list_len = show_pred(dir_img,
                                                      dir_img_with_points,
                                                      list_video_names,
                                                      number_of_cadrs,
                                                      need_count,
                                                      form_data1,
                                                      pause_on_marked_up_photo)

    # получение предсказания
    res_table = control_predict(data_all, list_coords_count, list_len)

    # в конце удаляем папку с нарезанными кадрами (vid_img)
    try:
        shutil.rmtree(dir_img)
    except:
        pass

    return res_table


if __name__ == "__main__":

    dir_vid = 'folders_for_test/videos/'
    dir_img_with_points = 'folders_for_test/imgs/'

    # try:
    #     shutil.rmtree(dir_vid)
    # except:
    #     pass

    try:
        shutil.rmtree(dir_img_with_points)
    except:
        pass

    if not os.path.isdir(dir_vid):
        os.mkdir(dir_vid)
    if not os.path.isdir(dir_img_with_points):
        os.mkdir(dir_img_with_points)

    form_function = form_data1

    res_table = classification(form_function,
                               dir_vid,
                               dir_img_with_points,
                               pause_on_marked_up_photo=None)
