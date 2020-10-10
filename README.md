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

## Описание классификатора и инструкция по использованию
<details>
  <summary>Раскрыть</summary><br/>
  
  ## Описание
  
  ## Использование
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
  clf.class_names 
  >>> ['2НДФЛ', '3НДФЛ', 'Форма банка', 'Неизвестный документ']
  clf.pred_labels
  >>>

  ```
   Допускается использование классификатора на уже загруженных и обработанных изображениях, 
   полученных методом _load_process_images.
   Результат кода будет идентичен вызову метода predict.
   
  ```python
  import tensorflow as tf
  import TaxFormClassificator
  clf = TaxFormClassificator.TaxFormClf()

  # Загружаем изображения
  proc_imgs, texts, img_names = clf._load_process_images('folder_with_images')
  # Классифицируем
  predictions = clf._form_predictions(proc_imgs, texts, img_names)
  ```
</details>








