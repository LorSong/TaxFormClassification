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

## Описание классификатора документов о доходе ФЛ
<details>
  <summary>Раскрыть</summary><br/>
  
  
  Использование. 
  Из за особенностей загрузки моделей, необходимо импортировать модуль tensorflow as tf
  
  ```python
  import tensorflow as tf
  import TaxFormClassificator
  # Иницируйте классификатор. На этой стадии он загрузит tf модель
  clf = TaxFormClassificator.TaxFormClf()
  # Вызовите метод predict указав папку с изображениями
  clf.predict('folder_with_images')
  ```
 
</details>
