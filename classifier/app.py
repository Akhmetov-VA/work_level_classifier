import datetime
import json
import pickle
from datetime import date

import joblib
import pandas as pd
import streamlit as st

from classifier.app_utils import (
    extract_workplace,
    pg_save,
    search_regexp,
    select_workplaces,
    visualize,
)
from classifier.postgres import PGInstance
from classifier.utils import BERTEmbExtractor

st.set_page_config(layout="wide")


@st.cache_resource
def load_pg_instance():
    pg_instance = PGInstance()
    return pg_instance


# Load the pre-trained model
@st.cache_resource
def load_model():
    file = "models/pipelines.joblib"
    with open(file, "rb") as f:
        pipes = joblib.load(f)
    return pipes


@st.cache_resource
def load_data_transform():
    file = "models/data_prepare.joblib"
    with open(file, "rb") as f:
        data_prepare = joblib.load(f)
    return data_prepare


@st.cache_resource
def load_label_encoder():
    file = "models/label_encoders.joblib"
    with open(file, "rb") as f:
        LE = joblib.load(f)
    return LE


@st.cache_data
def load_constants():
    from classifier.constants import countries, districts, educations, regions

    labels = ["Label 1", "Label 2", "Label 3"]
    return countries, districts, regions, educations, labels


@st.cache_data
def load_dicts():
    additional_data = {
        "Номер": None,
        "ИНН": None,
        "Вид экономической деятельности, ОКВЭД": None,
        "Доп вид экономической деятельности_1": None,
        "Доп вид экономической деятельности_2": None,
        "Доп вид экономической деятельности_3": None,
        "Уставный капитал, тип": None,
        "Уставный капитал, сумма": None,
        "Тип по ОКОГУ": None,
        "Среднесписочная численность сотрудников": None,
        "Категория из реестра СМП": None,
        "Сумма уплаченных налогов за 2020": None,
    }

    input_data = {
        "Пол": None,
        "Дата рождения": None,
        "Страна проживания": None,
        "Федеральный округ": None,
        "Регион": None,
        "Уровень образования": None,
        "Место работы": None,
        "Наименование текущей должности": None,
    }
    return additional_data, input_data


@st.cache_data
def load_regexp():
    with open("models/regexp_pattern.json") as file:
        regex_for_three_gov_fields = json.loads(file.read())

    matching_three_gov_fields = {
        "foiv": "Федеральные государственные органы, в т.ч. федеральная государственная служба",
        "roiv": "Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ",
        "mestnie": "Органы местного самоуправления (ОМСУ)",
        "foiv roiv": "Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ",
        "mestnie foiv": "Органы местного самоуправления (ОМСУ)",
        "mestnie foiv roiv": "Органы местного самоуправления (ОМСУ)",  # вот тут вопрос!
        "mestnie roiv": "Органы местного самоуправления (ОМСУ)",
    }

    return matching_three_gov_fields, regex_for_three_gov_fields


# Function to preprocess input data
def preprocess_input(data, data_transform):
    data["Дата рождения"] = pd.to_datetime(data["Дата рождения"])
    data["age"] = pd.Timestamp("now").year - data["Дата рождения"].dt.year

    data = data_transform.transform(data)
    return data


# Function to make predictions
def predict(model, data):
    return model.predict(data)


@st.cache_data
def predict_upload_df(
    uploat_df, _input_data, _additional_data, _data_transform, _pipes, _pg_instance, _LE
):
    preds1 = []
    preds2 = []

    for i, (workplace, jobname) in uploat_df[
        ["Место работы", "Наименование текущей должности"]
    ].iterrows():
        _input_data["Место работы"] = workplace
        _input_data["Наименование текущей должности"] = jobname

        _additional_data = extract_workplace(workplace, _additional_data)
        _input_data = _input_data | _additional_data
        input_df = pd.DataFrame([_input_data])

        processed_data = preprocess_input(input_df.copy(), _data_transform)
        prediction1 = predict(_pipes[0], processed_data)
        prediction2 = predict(_pipes[1], processed_data)

        # pg_save(_pg_instance, _input_data, _LE, prediction1, prediction2)

        preds1.append(_LE[0].classes_[prediction1])
        preds2.append(_LE[1].classes_[prediction2])

    uploat_df["Сфера деятельности по Классификатору ФОИР"] = preds1
    uploat_df["Карьерная ступень по Классификатору ФОИР"] = preds2

    return uploat_df


# Streamlit app
def main():
    additional_data, input_data = load_dicts()
    pipes = load_model()
    data_transform = load_data_transform()
    LE = load_label_encoder()
    pg_instance = load_pg_instance()
    matching_three_gov_fields, regex_for_three_gov_fields = load_regexp()

    countries, districts, regions, educations, labels = load_constants()

    st.title(
        "Система автоматического определения карьерных параметров (сфера деятельности, карьерная ступень)"
    )  # TODO: ввести норм название и инструкцию пользователя

    with st.sidebar:
        uploaded_file = st.file_uploader("Выберите excel-файл")

    if uploaded_file:
        uploat_df = pd.read_excel(uploaded_file)
        st.write("Загружен следующий файл:")
        uploat_df = uploat_df.dropna()
        # uploat_df = uploat_df.sample(10)
        # st.write(uploat_df.columns)

        st.write(uploat_df)

        # извлечем с помощью регулярки метки
        uploat_df = search_regexp(
            uploat_df, regex_for_three_gov_fields, matching_three_gov_fields
        )

        st.write("Результат работы регулярных выражений:")
        st.write(uploat_df)

        st.write("Результат прогноза модели:")

        uploat_df = predict_upload_df(
            uploat_df,
            input_data,
            additional_data,
            data_transform,
            pipes,
            pg_instance,
            LE,
        )

        st.write(uploat_df)

        visualize(uploat_df.copy())

    else:
        # Input form
        st.header("Данные о пользователе")
        gender = st.radio("Пол", ["Мужской", "Женский"], index=None)
        max_date = datetime.datetime.now().year - 18
        dob = st.date_input(
            "Дата рождения",
            date(1984, 1, 1),
            format="DD.MM.YYYY",
            min_value=date(1900, 1, 1),
            max_value=date(max_date, 1, 1),
        )

        country = st.selectbox("Страна проживания", countries, index=None)
        if country == "Другое":
            country = st.text_input("Введите другую страну...")

        district = st.selectbox("Федеральный округ", districts, index=None)
        if district == "Другое":
            district = st.text_input("Введите другой федеральный округ...")

        region = st.selectbox("Регион", regions, index=None)
        if region == "Другое":
            region = st.text_input("Введите другой регион...")

        education = st.selectbox("Уровень образования", educations, index=None)
        if education == "Другое":
            education = st.text_input("Введите другой регион...")

        jobname = st.text_input("Наименование текущей должности", None)

        workplace = st.text_input("Место работы", None)

        if workplace and jobname:
            additional_data = select_workplaces(workplace, additional_data)

            if st.button("Вывести результаты"):
                # Prepare input data
                input_data = {
                    "Пол": gender,
                    "Дата рождения": dob,
                    "Страна проживания": country,
                    "Федеральный округ": district,
                    "Регион": region,
                    "Уровень образования": education,
                    "Место работы": workplace,
                    "Наименование текущей должности": jobname,
                }
                input_data = input_data | additional_data

                input_df = pd.DataFrame([input_data])

                st.write(input_df)

                # Preprocess input data
                processed_data = preprocess_input(input_df.copy(), data_transform)

                # Display the processed data
                # st.subheader("Processed Input Data")
                # st.write(processed_data)

                # Make prediction
                prediction1 = predict(pipes[0], processed_data)
                prediction2 = predict(pipes[1], processed_data)

                # Display prediction
                st.subheader("Сфера деятельности по Классификатору ФОИР")
                st.write(LE[0].classes_[prediction1])

                st.subheader("Карьерная ступень по Классификатору ФОИР")
                st.write(LE[1].classes_[prediction2])

                print(LE[0].classes_[prediction1][0])

                # write input data into database
                pg_save(pg_instance, input_data, LE, prediction1, prediction2)

        if st.button("Показать последние запросы"):
            pg_instance.cursor.execute("select * from queries")
            data = pg_instance.cursor.fetchall()
            st.write(pd.DataFrame(data))


if __name__ == "__main__":
    main()
