import logging
import time

# импорт модуля распознования
import main

logging.basicConfig(filename="exec_logs.log",
                    filemode="w",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Классификатор финально обучен на данном формате
form_function = main.form_data1   # функция формирования numpy маасивов

# Блок распознавания
exec_time = time.time()
main.classification(form_data1=form_function,
                    pause_on_marked_up_photo=None)
logging.info(f"Prediction_time : {time.time() - exec_time}")