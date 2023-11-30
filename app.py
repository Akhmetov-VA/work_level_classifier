import streamlit as st
import pandas as pd
import pickle
from utils import BERTEmbExtractor
from app_utils import select_workplaces
import joblib
import datetime
from datetime import date

# Load the pre-trained model
@st.cache_resource
def load_model():
    file = "pipelines.joblib"
    with open(file, 'rb') as f:
        pipes = joblib.load(f)
    return pipes

@st.cache_resource
def load_data_transform():
    file = 'data_prepare.joblib'
    with open(file, 'rb') as f:
        data_prepare = joblib.load(f)
    return data_prepare

@st.cache_resource
def load_label_encoder():
    file = 'label_encoders.joblib'
    with open(file, 'rb') as f:
        LE = joblib.load(f)
    return LE

@st.cache_data
def load_constants():
    countries = ['Россия', 'Казахстан', 'Украина', 'Киргизия', 'Беларусь', 'Германия', 'Узбекистан', 'Латвия', 
                'Туркменистан', 'Армения', 'Молдова', 'Азербайджан', 'Таджикистан', 'США', 'Великобритания', 
                'Эстония', 'Австралия', 'Турция', 'Канада', 'Италия', 'Австрия', 'Американское Самоа', 'Болгария', 
                'Израиль', 'Сербия', 'Грузия', 'Алжир', 'Албания', 'Индия', 'Франция', 'Вьетнам', 'Польша', 'Швейцария', 
                'Чехия', 'Колумбия', 'Испания', 'Китай', 'ОАЭ', 'Аландские о-ва', 'Македония', 'Сирия', 'Нидерланды', 
                'Бельгия', 'Литва', 'Швеция', 'Ангилья', 'Египет', 'Греция', 'Марокко', 'Монголия', 'Венгрия', 'Финляндия', 
                'Ирак', 'Норвегия', 'Португалия', 'Япония', 'Таиланд', 'Венесуэла', 'Афганистан', 'Ливан', 'Мексика', 
                'Судан', 'Иран', 'Люксембург', 'Аргентина', 'Кипр', 'Пакистан', 'Перу', 'Индонезия', 'Конго - Браззавиль',
                'Бразилия', 'Палестинские территории', 'Румыния', 'Словакия', 'Шри-Ланка', 'Камбоджа', 'Тунис', 'Черногория',
                'Боливия', 'Танзания', 'Республика Корея', 'Хорватия', 'Босния и Герцеговина', 
                'Специальный административный регион Китая Гонконг', 'Йемен', 'Мали', 'Чили', 'Эквадор', 'Иордания', 
                'Эфиопия', 'Дания', 'Катар', 'Кот-д’Ивуар', 'Куба', 'Мальта', 'Исландия', 'Словения', 'Сингапур',
                'Камерун', 'Республика Южная Осетия', 'Замбия', 'Руанда', 'Чад', 'Сомали', 'Федеративные Штаты Микронезии',
                'Республика Абхазия', 'Бенин', 'Бурунди', 'Новая Зеландия', 'Антарктида', 'Ливия', 
                'Доминиканская Республика', 'Сен-Бартелеми', 'Коста-Рика', 'Гана', 'Ирландия', 'острова Питкэрн',
                'Гвинея', 'Мозамбик', 'ЮАР', 'Кувейт', 'Нигерия', 'Гамбия', 'Гондурас', 'Восточный Тимор', 'КНДР', 
                'Уругвай', 'Антигуа и Барбуда', 'Тайвань', 'Буркина-Фасо', 'Гаити', 'Другое']
    districts = ['Центральный ФО', 'Приволжский ФО', 'Северо-Западный ФО', 'Уральский ФО',
                 'Сибирский ФО', 'Южный ФО', 'Дальневосточный ФО', 'Северо-Кавказский ФО', 'Другое']
    regions = ['Москва', 'Санкт-Петербург', 'Московская', 'Свердловская', 'Краснодарский', 'Татарстан', 'Самарская',
               'Башкортостан', 'Новосибирская', 'Нижегородская', 'Тюменская', 'Челябинская', 'Ростовская', 'Красноярский',
               'Ханты-Мансийский Автономный округ - Югра', 'Пермский', 'Иркутская', 'Ставропольский', 'Воронежская', 
               'Ямало-Ненецкий', 'Саратовская', 'Ленинградская', 'Кемеровская', 'Белгородская', 'Волгоградская', 'Омская',
               'Мурманская', 'Дагестан', 'Приморский', 'Удмуртская', 'Томская', 'Хабаровский', 'Крым', 'Ярославская', 
               'Калининградская', 'Калужская', 'Алтайский', 'Липецкая', 'Кировская', 'Оренбургская', 'Смоленская', 'Рязанская', 
               'Тульская', 'Бурятия', 'Пензенская', 'Севастополь', 'Саха ( Якутия )', 'Архангельская', 'Вологодская', 
               'Чувашская', 'Коми', 'Владимирская', 'Тверская', 'Ивановская', 'Мордовия', 'Новгородская', 'Ульяновская',
               'Кабардино-Балкарская', 'Орловская', 'Забайкальский', 'Курская', 'Костромская', 'Карачаево-Черкесская',
               'Астраханская', 'Брянская', 'Сахалинская', 'Камчатский', 'Амурская', 'Карелия', 'Тамбовская', 'Хакасия',
               'Марий Эл', 'Северная Осетия - Алания', 'Тыва', 'Чеченская', 'Псковская', 'Курганская', 'Алтай', 'Адыгея', 
               'Калмыкия','Еврейская', 'Магаданская', 'Ненецкий', 'Ингушетия', 'Чукотский', 'Другое']
    educations = ['Высшее, специалитет, магистратура', 'Два и более высших образований', 'Высшее, бакалавриат',
                  'Кандидат наук', 'Высшее образование и MBA', 'Незаконченное высшее', 
                  'Студент выпускного курса специалитета/магистратуры', 'Среднее профессиональное', 'Доктор наук',
                  'Среднее общее', 'Другое']
    labels = ["Label 1", "Label 2", "Label 3"]
    return countries, districts, regions, educations, labels

# Function to preprocess input data
def preprocess_input(data, data_transform):
    data['Дата рождения'] = pd.to_datetime(data['Дата рождения'])
    data['age'] = pd.Timestamp('now').year - data['Дата рождения'].dt.year  

    data = data_transform.transform(data)
    return data

# Function to make predictions
def predict(model, data):
    return model.predict(data)

# Streamlit app
def main():
    pipes = load_model()
    data_transform = load_data_transform()
    LE = load_label_encoder()
    countries, districts, regions, educations, labels = load_constants()
    
    st.title("ML Model Prediction App")

    # Input form
    st.header("Данные о пользователе")
    gender = st.radio("Пол", ["Мужской", "Женский"], index=None)
    max_date = datetime.datetime.now().year - 18
    dob = st.date_input("Дата рождения", date(1984, 1, 1), format="DD.MM.YYYY", min_value=date(1900, 1, 1), max_value=date(max_date, 1, 1))
    
    country = st.selectbox('Страна проживания', countries, index=None)
    if country == 'Другое':
        country = st.text_input("Введите другую страну...")
    
    district = st.selectbox("Федеральный округ", districts, index=None)
    if district == "Другое": 
        district = st.text_input("Введите другой федеральный округ...")
        
    region = st.selectbox('Регион', regions, index=None)
    if region == "Другое": 
        region = st.text_input("Введите другой регион...")
    
    education = st.selectbox("Уровень образования", educations, index=None)
    if education == "Другое": 
        education = st.text_input("Введите другой регион...")
    
    workplace = st.text_input("Место работы", None)
    jobname = st.text_input('Наименование текущей должности', None)
    
    if workplace and jobname:
        additional_data = {'Номер': None,
                    'ИНН': None,
                    'Вид экономической деятельности, ОКВЭД': None,
                    'Доп вид экономической деятельности_1': None,
                    'Доп вид экономической деятельности_2': None,
                    'Доп вид экономической деятельности_3': None,
                    'Уставный капитал, тип': None,
                    'Уставный капитал, сумма': None,
                    'Тип по ОКОГУ': None,
                    'Среднесписочная численность сотрудников': None,
                    'Категория из реестра СМП': None,
                    'Сумма уплаченных налогов за 2020': None,       
                    }
                        
        additional_data = select_workplaces(workplace, additional_data)
        
        
        if st.button("Вывести результаты"):
            # Prepare input data
            input_data = {
                "Пол": gender,
                "Дата рождения": dob,
                'Страна проживания': country,
                "Федеральный округ": district,
                "Регион": region,
                'Уровень образования': education,
                "Место работы": workplace,
                'Наименование текущей должности': jobname,
            }
            input_data = input_data | additional_data
            
            input_df = pd.DataFrame([input_data])
            
            # additional_df = pd.DataFrame([additional_data])
            
            st.write(input_df)
            # st.write(additional_df)

            # Preprocess input data
            processed_data = preprocess_input(input_df.copy(), data_transform)

            # Display the processed data
            st.subheader("Processed Input Data")
            st.write(processed_data)

            # Make prediction
            prediction1 = predict(pipes[0], processed_data)
            prediction2 = predict(pipes[1], processed_data)

            # Display prediction
            st.subheader('Сфера деятельности по Классификатору ФОИР')
            st.write(LE[0].classes_[prediction1])
            
            st.subheader("Карьерная ступень по Классификатору ФОИР")
            st.write(LE[1].classes_[prediction2])

if __name__ == "__main__":
    main()
