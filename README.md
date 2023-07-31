<a><img width=70% src="https://github.com/amirazamatov/Donor_Search/blob/master/DonorSearch.png"></a><br><br>

# Распознование текста с фотографий медицинских документов

**Заказчик**: сообщество доноров крови **“DonorSearch”**.

Заказчику необходимо ML-решение для автоматического извлечения информации с фотографий медицинских документов.

**Цель проекта**: подготовить рабочее решение, которое позволит извлекать данные о донации крови из отсканированных или сфотографированных справок (форма 405), в частности:

- полную дату донации
- тип донации
- вид донации

Для работы нам было предоставлено 15 фотографий  и соответствующие им табличные данные, поэтому обучение нейронных сетей на представленных данных было невозможным. 

**Основные этапы проекта:**

1. Детекция табличных данных
2. Предобработка фотографий
3. Детекция и распознование текста с таблиц

# Выводы

В ходе работы нами разработан микросервис, который принимает изображения и выдаёт в ответ распознанные табличные данные с донациями крови в формате json. Полное решение можно посмотреть в [тетрадке.](https://github.com/amirazamatov/Donor_Search/blob/master/detection_recognition_scan_foto_medicaldocs.ipynb)

**Для детекции были опробованы два подхода**:

- детекция таблиц методами OpenCV. Это хороший вариант решения, который позволил бы избежать использования в коде сложных продуктов, но у него выявилось несколько недостатков: он плохо обрабатывает фото с артефактами (например, печатями), что требует написания дополнительного кода и не работает с некачественными, размытыми фотографиями.
- детекция таблиц с помощью обученной нейронной сети FAST RCNN. Нейронная сеть хорошо находит таблицы, но у нее есть проблемы с границами на фото с искаженной перспективой и мятых листах. Эту проблему легко удалось решить с помощью расширения отступов от границ бокса.

**Для предобработки фотографий были опробованы следующие инструменты:**

- методы OpenCV, включающие в себя повороты изображений, цветокоррекцию и выравнивание. Этот подход оказался очень эффективным.
- автоэнкодеры. В проекте не представлены из-за большого размера кода. Автоэнкодеры с латентным пространством не использовались, т.к. у нас недостаточно документов на русском языке, чтобы по ним можно было качественно обучить нейронную сеть. Поэтому использовалось прямое кодирование/декодирование изображений с помощью одного слоя свертки и пуллинга

**Распознавание текста**

- рассматривались  две библиотеки: pytesseract и easyocr. Одним из существенных минусов pytesseract является то, что модель очень плохо распознает текст на фотографиях с ненадлежащим качеством. Это критичный момент, т.к. у нас в выборке таких фото может быть много. Несомненным преимуществом easyocr перед другими моделями для распознавания еще и является то, что в нее встроены такие продвинутые алгоритмы детекции текста как CRAFT, что окончательно определило выбор в пользу этой библиотеки.

**В целом, не используя сложных инструментов, нам удалось достигнуть точности в 72%. Основная проблема заключается в том, что при искажении перспективы, размытии или деформации справки, EASYOCR начинает плохо считывать данные, что приводит у ухудшению результатов работы алгоритма. Следующим этапом для развития проекта может стать приведение фотографий в надлежащее качество с написанием собственных сетей на базе архитектур GAN и автоэнкодеров.**



# Микросервис для обработки медицинских справок

Репозиторий содержит сервис, который принимает изображения и выдаёт в ответ информацию об изображении.

## Cодержимое репозитория
- _requirements.txt_ - список python библиотек, необходимых для запуска сервиса. Чтобы установить необходимые пакеты, выполните команду
```
    pip install -r requirements.txt
```

- _src/app.py_ - основной скрипт репозитория, который запускает веб-сервер. Чтобы воспользоваться этим файлом, запустите его через вашу IDE (PyCharm, VSCode) или командную консоль.

- _src/model.py_ - файл содержит функцию _process_image_, которая принимает на вход путь до файла изображения, обрабатывает его и выдаёт ответ веб-серверу.

- _src/tmp_ - временная папка, куда будут сохраняться изображения, полученные сервисом.

  
## Как протестировать

1. Склонируйте репозиторий на локальный компьютер;
2. **В папку src необходимо поместить файл с FAST RCNN. Скачать его нужно по [ссылке](https://disk.yandex.ru/d/q8A7ifUOdkWu7w)**.
3. Для сборки образа необходимо выполнить команду
```
  docker build --tag ocr:0.1 .
```

4. Для запуска контейнера
```
  docker run --rm -d -p 8010:8000 --name ocr ocr:0.1
```

после выполнения этой команды сервис будет доступен на порту 8010

5. Перейдите по ссылкe в последней строке сообщения. В поле выбора файла выберите любой локальный файл с вашего компьютера и отправьте его.Результат обработки изображения возвращается пользователю как ответ на его запрос.

6. Для остановки запущенного контейнера выполните команду
```
  docker stop ocr
```
 


