# TaxFormClassification

Тестовое задание УБРиР.
Основная работа: создание модели способной классифицировать формы документов о доходах ФЛ. (2-НДФЛ, 3-НДФЛ, банковская форма или обнаружить некорректный документ)

## Описание файлов
<details>
  <summary>Раскрыть</summary><br/>
  
  1. 1_SQL_queries.txt - Текстовый документ с SQL запросами к первому заданию
  2. 2_ClassicML_DefaultDet.ipynb - Блокнот с основными шагами по выполнению задания 2.
  3. TaxFormClassificator.py - Скрипт выполняющий классификацию документов.
  4. model - архив с обученной tensorflow моделью, используемой классификатором
  5. requirements.txt - используемые библиотеки
  
  4. test_images - 5 тестовых изображений.
  
</details>

## Описание классификатора
<details>
  <summary>Раскрыть</summary><br/>
  
  Модель
</details>
    
    
## Инструкция по использованию
 <details>
   <summary>Раскрыть</summary><br/>
   Подготовка. Убедитесь, что у вас установлены необходимые python библиотеки указанные в requirements.txt.
   В особенности:
      **tensorflow** > 2.0.0 (лучше 2.3.0)
      **tesserocr** (вместе с tesseract, который должен установится по умолчанию вместе с tesserocr)
      **pdf2image**
      **fuzzywuzzy** (используется для сравнения текста)
   
   
   
   Так как tesseract выполняет OCR достаточно медленно, обработка одного изображения может занимать до 20 секунд

   ```python
   # Из за особенностей загрузки моделей, необходимо импортировать модуль tensorflow as tf
   import tensorflow as tf
   import TaxFormClassificator
   # Иницируйте классификатор. На этой стадии он загрузит tf модель
   clf = TaxFormClassificator.TaxFormClf()
   # Вызовите метод predict указав папку с изображениями
   predictions = clf.predict('folder_with_images')

   predictions
   >>> {filename_1.jpg: 'НДФЛ2', filename_2.jpg: 'НДФЛ3'}
   ```

   После выполнения метода predict, также сохраняются дополнительные атрибуты

   ```python
   # Названия классов
   clf.class_names 
   >>> ['2НДФЛ', '3НДФЛ', 'Форма банка', 'Неизвестный документ']
   
   # Предсказанные классы, соответствующие индексам в class_names. 
   # Порядок соответствует clf.img_names аттрибуту
   clf.pred_labels
   >>> array([1, 0, 1], dtype=int64)
   
   # 2d array с предсказанными вероятностями. 
   # Трансформируется в pred_labels путем np.argmax(probas, axis=1)
   clf.pred_probas
   # Аналогично, отдельно для CNN и OCR
   clf.cnn_probas
   clf.ocr_probas 
   
   # Лист с проведенными поворотами изображений (0, 90, 180, 270)
   clf.rotations
   >>> [270, 0, 0]
   
   # List с обработанными np.array изображениями
   clf.proc_images
   # List с полученными текстами, string
   clf.texts
   # Список имен файлов
   clf.img_names
   >>> ['12.png', '71.png', '9.png']

   ```
   Допускается использование классификатора на уже загруженных и обработанных изображениях, 
   полученных методом _load_process_images.
   Результат кода будет идентичен вызову метода predict.

   ```python
   import tensorflow as tf
   import TaxFormClassificator
   clf = TaxFormClassificator.TaxFormClf()

   # Загружаем изображения
   images, img_names = clf._read_images_from_folder('folder_with_images')
   # Обработка (возможен некорректный поворот)
   proc_imgs = self._preprocess_images(images, img_names)
   # Извлечение текста и исправление поворота
   proc_imgs, texts = self._extract_text_fix_orient(proc_imgs, img_names)
        
   # Классифицируем
   predictions = clf._form_predictions(proc_imgs, texts, img_names)
   ```

</details>
    









