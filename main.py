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
import time
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import pandas as pd

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
    '''
    :param path:
    :return:
    nnnknknknknknknknknknknknknknknknkknknkknknknknknknk
    '''

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
            cur_video = cur_video.replace(".", "")  # .replace(".", "") исключает точки из названия видео
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

    return list_video_names, number_of_cadrs


############################################################
# функция выделения признаков из нужных кадров ОДНОГО видео
############################################################

def show_pred(dir_source, list_video_names,number_of_cadrs, need_count, function):
    '''
    Описание
    '''
    

    data_class = np.array([])   # массив данных для классификатора
    data_NoR = []               # массив количества повторов на всех видео

#........................................Цикл формирования данных по всем видео 
    for video_name in list_video_names:
        # листы для классификации
        list_bbox   = []
        list_coords = []

        # листы для подсчета 
        list_bbox_count     = []
        list_coords_count   = []        
        

        # !!! Включить интерактивный режим для анимации
        plt.ion()

        video_name = video_name.replace(".", "")  # .replace(".", "") исключает точки из названия видео

        # если выбран режим подсчтета количества упражнений
        if need_count: 
#........................................Цикл формирования данных с кадров одного видео

            print(number_of_cadrs, type(number_of_cadrs))
            for counter in range(0, max(number_of_cadrs), 10):   # когда несколько доллжно быть [], ведь number_of_cadrs лист, что за бред
            
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
                print(count,'.....cont......................................................................................................................................................')
                count = 0

                # получение координат для классификации
                list_bbox_count.append(upscale_bbox)
                list_coords_count.append(pred_coords)
            
            # получение координат для подсчета
            list_bbox =  list_bbox_count[0:10]
            list_coords =  list_coords_count[0:10]

            print('list_bbox_count:', list_bbox_count)
            print('list_coords_count', list_coords_count)
            print('list_bbox', list_bbox)
            print('list_coords', list_coords)

        else: # если НЕ выбран режим подсчтета количества упражнений
        
             for counter in range(0, 100, 10):
             
                img_path = f'{dir_source}{video_name}_cadr_{str(counter)}.jpg'

                count = 0
                for try_step in range(10):
                    try:
                        upscale_bbox, pred_coords = predict(f'{dir_source}{video_name}_cadr_{str(counter + count)}.jpg')
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

#..................................данные для классификатора......................... 
        # Получаем сформированные данны текущего видео по данным для классификации
        data_video = function(list_bbox, list_coords)
        print(data_video)
        # Конкатинируем все видео и получаем общий массив с данными
        data_class = np.concatenate((data_class, data_video), axis=0) # ()
        print(data_class)
        
#..................................данные для счетчика......................... 
        # Получаем сформированные данны текущего видео по данным для классификации
        number_of_repet = function_count(list_coords_count)
        # Конкатинируем все видео и получаем общий массив с данными
        data_NoR.append(number_of_repet)       
        print('data_NoR:', data_NoR)

    return data_class, data_NoR, len(list_video_names)


def function_count (list_coords_count):
    list_y = []
    for cadr in list_coords_count: # cadr.shape = (1, 17, 2)
        list_y.append(cadr[0][11][1])   # берём координаты узла таза на наиболее вероятном человеке в кадре

    # считаем кол-во повторений
    check = True
    count = 0
    for i in range(1, len(list_y)):
        if list_y[i] - list_y[i - 1] < 0:
            check = True

        if check:
            if list_y[i] - list_y[i - 1] >= 0:
                check = False
                count += 1
    print(count)
    return count 





################################################################
# функция №1 преобразования признаков текущего видео в Data данные
################################################################

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
def control_predict(data_all, data_NoR, list_len):
    # нормализация данных
    data_norm = preprocessing.minmax_scale(data_all.T).T

    # ..........................подготовка данных для классификатора........................
    # изменение размерности
    data_all_reshape = data_norm.reshape(list_len, 10, 38)
    data_all_reshape.shape

    # добавление размерности для Conv слоёв
    x_test = data_all_reshape.reshape(list_len, 10, 38, 1)
    print(x_test.shape)

    # ..........................подготовка данных для счетчика........................
    number_of_cadrs = data_all.shape[0] / 38  # количество кадров
    data = data_all.reshape(int(number_of_cadrs), 19, 2)  # меняем размерность
    
    # вывод определенного класса
    names = ['приседания', 'подтягивания', 'отжимания']
    max_indent = 15

    prediction = exercise_recognizer.predict(x_test)  # получаем весь предикт

    df = pd.DataFrame(columns=['type of exercise', 'number of repetitions'])

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
            {'type of exercise': pred, 'number of repetitions' : data_NoR[i]})
#       df.loc[f'Net_{count_NN}'] = pd.Series({'batch':batch_now,'learnR':learnR_now,'hidden':hidden_now,'Time':time_point,'Loss':lossTot,'Acc':accuracy})
    window = tk.Tk()
    window.geometry('500x500')
    window.title("Таблица", )

    lbl = tk.Label(window, text=df, font=("Arial Bold", 25))
    lbl.grid(column=0, row=20)

    window.mainloop()


####################################
# обобщение всех глобальных функций
####################################
def classification(form_data1, min_cadrs_value=95):
    # получаем путь к директории с видео
    dir_vid = filedialog.askdirectory(title = 'Выберите директорию с видео')
    dir_vid = f'{dir_vid}/'

    # на основе него задаем путь для сохранения картинок
    dir_img = 'dir_img/'  # не забывать про /
    #    dir_img = f'{dir_vid[:-8]}{dir_img}/'

    print(dir_img)

    # повторный запуск mkdir с тем же именем вызывает FileExistsError,
    # вместо этого запустите:
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)

    # нарезка видео и сохранение кадров
    list_video_names, number_of_cadrs = save_frames(dir_vid, dir_img, min_cadrs_value)

    # Получение из кадров видео nampy массива
    data_all, data_NoR, list_len = show_pred(dir_img, list_video_names, number_of_cadrs, need_count = True, function = form_data1)

    # получение предсказания
    control_predict(data_all, data_NoR, list_len)

    # в конце удаляем папку с нарезанными кадрами (vid_img)
    shutil.rmtree(dir_img)