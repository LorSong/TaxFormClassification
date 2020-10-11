# TaxFormClassification

Тестовое задание УБРиР.
Основная работа: создание python модуля способного классифицировать формы документов о доходах ФЛ. (2-НДФЛ, 3-НДФЛ, банковская форма или обнаружить некорректный документ)

## Описание файлов
<details>
  <summary>Раскрыть</summary><br/>
  
  1. 1_SQL_queries.txt - Текстовый документ с SQL запросами к первому заданию
  2. 2_ClassicML_DefaultDet.ipynb - Блокнот с основными шагами по выполнению задания 2.
  3. TaxFormClassificator.py - Python модуль выполняющий классификацию документов.
  4. model - архив с обученной tensorflow моделью, используемой классификатором
  5. requirements.txt - используемые библиотеки
  6. test_images - 5 тестовых изображений.
  
</details>

## Описание классификатора
<details>
  <summary>Раскрыть</summary><br/>
  Классификатор представляет собой Python класс, который вы можете импортировать как модуль и использовать
  следуя инструкции ниже.
  
  На изображениях высокого качества ошибки маловероятны. В случаях низкого разрешения, скошенной перспективы и размытости,
  точность снижается. В связи с невысоким количество тестовых изображений, я могу дать лишь приблизительные показания точности.
  На сканированных документах - свыше 95%
  На неровно сфотографированных документах - около 75%
  
  Принцип работы:
  Классификация происходит путем комбинации двух методов: сравнение текста и использование нейронной сети. Оба метода дают
  ошибки разного рода и могут корректировать друг друга.<br/>
  Перед применением этих методов изображения проходят подготовительные этапы. Они включают в себя:
  * Исправление перспективы. Если документ на изображении размещен под углом и имеет фон, программа пытается повернуть его, 
  чтобы получить вид сверху. Этот шаг не всегда успешен, если границы листа и фона нечеткие, в этих случаях документ остается
  неизменным
  * Исправление наклона. Программа находит минимальный прямоугольник, в котором располагается весь текст и вычисляет угол его
  поворота. Поворот с этим углом применяется на все изображение.
  * Исправление тени путем выравнивания среднего значения цвета по всему изображению. Также происходит увеличение контрастности.
  * Исправление поворота происходит во время процесса распознавания текста.
  * Распознавание текста происходит на увеличенных изображениях (4000 пикселей сторона), а нейронная сеть уменьшает изображения до
  224х224 пискелей.
  
  Классификаторы:
  * Сравнение текста
    После подготовки изображения из документа извлекается текст путем применения программы tesseract. Полученный текст сравнивается
    с заранее заданными ключевыми словами из документов путем использования расстояния Левенштейна. Среднее значение результатов для 
    каждого класса трансформируются в вероятности принадлежности к классу.
   * Нейронная сеть
    Для классификации была обучена последовательная сверточная нейронная сеть с 6 обучаемыми слоями. Тренировочный сет был дополнен
    несколькими десятками изображений взятых из интернета и аугментирован. Был добавлен случайный шум, поворот и зум. Также для
    предотвращения оверфита был сформирован небольшой валидационный сет (30 изображений).
    Архитектура модели:
    Тип слоя   Кол-во фильтров/нейронов  Размер окна    Шаги    Активация
    Convolutional         16                 12          3        relu
    Convolutional         32                  7          2        relu
    Convolutional         64                  3          1        relu
    MaxPooloing
    Convolutional        128                  3          1        relu
    MaxPooloing
    Flatten
    Dropout(0.3)
    Dense                256                                      relu
    Dropout(0.3)
    Dense                 4                                      softmax          
  
  * Объединение результатов двух классификаторов
   Результаты объединяются используя специально подобранную схему.
   В случаях когда оба классификатора считают, что получен документ принадлижащий к одному из известных классов их результаты
   суммируются, давая больше веса текстовому классификатору. В случаях если текстовый классификатор считает, что документ принадлежит к неизвестному
   классу, используется предсказание нейронной сети, но только в том случае если она в нем уверена. И в случаях когда нейронная сеть 
   считает, что документ принадлежит к неизвестному, её вклад в общий прогноз занижается.
</details>
    
## Инструкция по использованию
<details> <summary>Раскрыть</summary><br/>
  
  Подготовка. Убедитесь, что у вас установлены необходимые python библиотеки указанные в requirements.txt. <br/>
  
  В особенности: <br/>
  * tensorflow > 2.0.0 (лучше 2.3.0)
  * tesserocr (вместе с tesseract, который должен установится по умолчанию вместе с tesserocr)
  * pdf2image
  * fuzzywuzzy (используется для сравнения текста)
  * cv2, numpy, scipy, PIL
   
   Так как tesseract выполняет OCR достаточно медленно, обработка одного изображения может занимать до 20 секунд

   ```python
   # Из за особенностей работы tensorflow, необходимо импортировать модуль tensorflow as tf
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
   proc_imgs = clf._preprocess_images(images, img_names)
   # Извлечение текста и исправление поворота
   proc_imgs, texts = clf._extract_text_fix_orient(proc_imgs, img_names)
        
   # Классифицируем
   predictions = clf._form_predictions(proc_imgs, texts, img_names)
   ```

</details>
    









