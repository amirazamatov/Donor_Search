import pandas as pd
import numpy as np
import easyocr
import re

easyocr = easyocr.Reader(['ru'])

BLOOD_CLASS = {
    'Цельная кровь': 'кр~ы9бо',
    'Тромбоциты': 'цф',
    'Плазма': 'пшиъ['
}

DONATION_TYPE = {
    'Безвозмездно': 'бвзоаеы06389',
    'Платно': 'п'
}

def easyocr_results(image):
    result = easyocr.readtext(image,
                              height_ths=0.9,
                              width_ths=0.9,
                              text_threshold=0.6,
                              contrast_ths=0.3,
                              min_size=1
                              )
    return result

# результаты easyocr переводим в списки, чтобы потом сделать таблицу
# если разница координат по высоте между значением и средним значением для этой строчки меньше 0.06, то это одна строчка
# если разница больше, то новая строка.
def easyocr_to_list(results_ocr):
    list_row = []
    table = []
    mean_height = results_ocr[0][0][2][1]
    for result in results_ocr:
        if (result[0][2][1] - mean_height) / mean_height < 0.06:
            list_row.append(result[1])
            mean_height =  (mean_height + result[0][2][1]) / 2
        else:
            table.append(list_row)
            list_row = []
            mean_height = result[0][2][1]
            list_row.append(result[1])
    table.append(list_row)
    return table

def easyocr_to_dataframe(list_of_rows):
    list_row = []
    table = []
    for row in list_of_rows:
        for donation_date in range(0, len(row), 3):
            try:
                list_row = []
                list_row.append(row[donation_date])
                list_row.append(row[donation_date + 1])
                if len(list_row) == 2:
                    table.append(list_row)
            except:
                pass
        if len(list_row) == 2:
            table.append(list_row)
    df = pd.DataFrame(table, columns=['date','type'])
    return df


def transform(string, dictionary):
    convert_value = 'Нужен ручной ввод'
    for key, values in dictionary.items():
        if any(((char in values) for char in string.lower())):
            convert_value = key
            break
    return convert_value


def transform_donation_date(value):
    for i in value:
        if i in ')(}{][|\/liI"':
            i_index = list(value).index(i)
            value = value[:i_index] + '' + value[i_index + 1:]
        if i in "OCGDQPПОСРocосqdp":
            i_index = list(value).index(i)
            value = value[:i_index] + '0' + value[i_index + 1:]
    value = "".join(value.split())
    if value[-4] in '.,':
        value += '0'
    if (len(value) == 9) and (re.match(r"\w{2}\W\w{6}", value) != None):
        value = value[:2] + '.' + value[3:5] + '.' + value[5:]
    if (len(value) == 9) and (re.match(r"\w{4}\W\w{4}", value) != None):
        value = value[:2] + '.' + value[2:4] + '.' + value[5:]
    if (len(value) == 8) and (re.match(r"\w{8}", value) != None):
        value = value[:2] + '.' + value[2:4] + '.' + value[4:]
    if (len(value) == 8) and (re.match(r"\w{2}\W\w{2}\W\w{2}", value) != None):
        value = value[:6] + '20' + value[6:]
    if (len(value) == 10) and (re.match(r"\w{2}\W\w{2}\W\w{4}", value) != None):
        value = value[:2] + '.' + value[3:5] + '.' + value[6:]
        value = value[:6] + '20' + value[8:]
    if (value[-4:-2] == '20') and value[-5] != '.':
        value = value[:-4] + '.' + value[-4:]
    if value[0] in '96ао':
        value = '0' + value[1:]
    if value[0] in '7':
        value = '1' + value[1:]
    if value[0] in '8BВ':
        value = '3' + value[1:]
    if (len(value) == 8) and (re.match(r"\w{3}\W\w{4}", value) != None):
        value = value[:2] + '.1' + value[2:]
    if value[3] in '496ао':
        value = value[:3] + '0' + value[4:]
    if (value[3] == '1') and (value[4] in '98653ао'):
        value = value[:4] + '0' + value[5:]
    if (value[3] == '1') and (value[4] in '7'):
        value = value[:4] + '1' + value[5:]
    if len(value) > 10:
        convert_value = ''
        dot_counter = 0
        for i in range(len(value)):
            if value[i].isalnum() == True:
                dot_counter = 0
                convert_value += value[i]
            if value[i].isalnum() == False:
                dot_counter += 1
                if dot_counter == 1:
                    convert_value += '.'
        value = convert_value
    if len(value) > 10:
        value = value[1:]

    return value

# Функция для сортировки таблицы по дате
def sort_date(dataframe):
    try:
        dataframe.iloc[:,1] =  pd.to_datetime(dataframe.iloc[:,1], format='%d.%m.%Y')
        dataframe.sort_values(by=dataframe.columns[1], inplace=True, ignore_index=True)
        dataframe.iloc[:,1] = dataframe.iloc[:,1].apply(lambda x: x.strftime('%d.%m.%Y'))
    except:
        pass
    return dataframe

def extract_data(image):
    example_result_easyocr = easyocr_results(image)
    result = easyocr_to_list(example_result_easyocr)
    df = easyocr_to_dataframe(result)
    df['digit_num'] = df['date'].apply(lambda x: [(i).isdigit() for i in x].count(True))
    df['row_lenth'] = df['date'].apply(lambda x: len(x))

    df = df.query('digit_num > 5 and row_lenth < 12')
    df['Класс крови'] = df.iloc[:, 1].apply(lambda x: transform(x[:3], BLOOD_CLASS))
    df['Дата донации'] = df.iloc[:, 0].apply(lambda x: transform_donation_date(x))
    df['Тип донации'] = df.iloc[:, 1].apply(lambda x: transform(x[1:], DONATION_TYPE))

    df.drop_duplicates(inplace=True)
    df = df.loc[:, ['Класс крови', 'Дата донации', 'Тип донации']]
    df = sort_date(df)

    return df