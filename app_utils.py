import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import dadata
from dadata import Dadata
import requests

import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from pathlib import Path

import re


ofdata_token = 'JrFCUBe4XNHhBYCy'
dadata_token = '41070744a4728056044eab22b9adb74beff1bfe9'

dadata = Dadata(dadata_token)

## Функции по загрузке запросов в БД

sql = """
INSERT INTO queries (gender, dob, country, district, region, education, workplace, jobname, inn, 
                    okwed_type, okwed_1, okwed_2, okwed_3, capital_type, capital_value, okogu_type, 
                    employee_cnt, smp_cat, taxes_2022, scope_work, сareer_stage) 
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
"""

def pg_save(pg_instance, input_data, LE, prediction1, prediction2):
    pg_instance.cursor.execute(sql, (input_data['Пол'], 
                                    input_data["Дата рождения"], 
                                    input_data['Страна проживания'],
                                    input_data['Федеральный округ'], 
                                    input_data['Регион'], 
                                    input_data['Уровень образования'], 
                                    input_data['Место работы'], 
                                    input_data['Наименование текущей должности'], 
                                    input_data['ИНН'], 
                                    input_data['Вид экономической деятельности, ОКВЭД'],
                                    input_data['Доп вид экономической деятельности_1'],
                                    input_data['Доп вид экономической деятельности_2'],
                                    input_data['Доп вид экономической деятельности_3'],
                                    input_data['Уставный капитал, тип'],
                                    input_data['Уставный капитал, сумма'],
                                    input_data['Тип по ОКОГУ'],
                                    input_data['Среднесписочная численность сотрудников'],
                                    input_data['Категория из реестра СМП'],
                                    input_data['Сумма уплаченных налогов за 2020'],
                                    str(LE[0].classes_[prediction1][0]),
                                    int(LE[1].classes_[prediction2][0])))



### Функции для обогащения данными по ИНН

def request_inn(company_inn):
    all_data_about_company = requests.get(f'https://api.ofdata.ru/v2/company?key={ofdata_token}&inn={company_inn}')
    return all_data_about_company


def extract_workplace(workplace, additional_data):
    fns_data = dadata.suggest("party", workplace)
    number_of_companies = len(fns_data)
    
    if number_of_companies >= 1:
        company_inn = fns_data[0]['data']['inn']
        
        all_data_about_company = request_inn(company_inn)
        
        additional_data = extract_add_data(all_data_about_company, additional_data)
    
    return additional_data

def extract_add_data(all_data_about_company, additional_data):
    # получаем все-все данные о компании:
    all_data_about_company = all_data_about_company.json()

    additional_data['ИНН'] = all_data_about_company['data'].get('ИНН')

    additional_data['Уставный капитал, тип'] = all_data_about_company['data']['УстКап'].get('Тип')

    additional_data['Уставный капитал, сумма'] = all_data_about_company['data']['УстКап'].get('Сумма')

    additional_data['Вид экономической деятельности, ОКВЭД'] = all_data_about_company['data']['ОКВЭД'].get('Наим')
    
    if len(all_data_about_company['data']['ОКВЭДДоп']) >= 1:
        additional_data['Доп вид экономической деятельности_1'] = all_data_about_company['data']['ОКВЭДДоп'][0].get('Наим')
        
    if len(all_data_about_company['data']['ОКВЭДДоп']) >= 2:
        additional_data['Доп вид экономической деятельности_2'] = all_data_about_company['data']['ОКВЭДДоп'][1].get('Наим')
        
    if len(all_data_about_company['data']['ОКВЭДДоп']) >= 3:
        additional_data['Доп вид экономической деятельности_3'] = all_data_about_company['data']['ОКВЭДДоп'][2].get('Наим')

    additional_data['Среднесписочная численность сотрудников'] = all_data_about_company['data'].get('СЧР')

    additional_data['Категория из реестра СМП'] = all_data_about_company['data']['РМСП'].get('Кат')

    additional_data['Сумма уплаченных налогов за 2020'] = all_data_about_company['data']['Налоги'].get('СумУпл')

    additional_data['Тип по ОКОГУ'] = all_data_about_company['data']['ОКОГУ'].get('Наим')
    return additional_data


def select_workplaces(workplace, additional_data):
    fns_data = dadata.suggest("party", workplace)

    # компаний ведь может быть несколько!
    number_of_companies = len(fns_data)

    # чтобы челу сориентироваться - дадим ему название и адрес регистрации компании:
    company_names = [fns_data[i]['value'] for i in range(number_of_companies)] # [fns_data[i]['data']['name']['full_with_opf'] for i in range(number_of_companies)]
    company_adresses = [fns_data[i]['data']['address']['unrestricted_value'] for i in range(number_of_companies)]

    company_names_and_adresses = []
    for i in range(len(company_names)):
        company_names_and_adresses.append(company_names[i] + ' (' + company_adresses[i] + ')')
        
    company_name = st.multiselect('Выберете компанию, подходящую под ваше название', company_names_and_adresses)
    
    if company_name:
        index = company_names_and_adresses.index(company_name[0])
        company_inn = fns_data[index]['data']['inn']

        all_data_about_company = request_inn(company_inn)
        
        additional_data = extract_add_data(all_data_about_company, additional_data)

        st.write(additional_data)
                
    return additional_data


## Функции для визуализации карьерной ступени

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

def visualize(df):
    logins_to_show_career_track = st.multiselect('Выберете логин, чью карьерную траекторию отображать:', 
                                                 df['Номер '].sort_values().unique(), 
                                                 default=df['Номер '].sort_values().unique()[0])
        
    fig = go.Figure()

    # for login in logins_to_show_career_track:
    df['Карьерная ступень по Классификатору ФОИР'] = df['Карьерная ступень по Классификатору ФОИР'].astype('int')
    df['Сфера деятельности по Классификатору ФОИР'] = df['Сфера деятельности по Классификатору ФОИР'].apply(lambda x: x[0]).astype('str')

    # fig = px.scatter(df[df['Номер '].isin(logins_to_show_career_track)], 
    #                 x="Начало периода должности", 
    #                 y="Карьерная ступень по Классификатору ФОИР", 
    #                 title='Визуализация уровней должностей',
    #                 color='Номер ',
    #                 symbol='Сфера деятельности по Классификатору ФОИР',
    #                 )
    
    # fig.add_scatter(df[df['Номер '].isin(logins_to_show_career_track)],
    #                          x="Начало периода должности", 
    #                          y="Карьерная ступень по Классификатору ФОИР",
    #                          symbol='Сфера деятельности по Классификатору ФОИР',
    #                         )
    for login in logins_to_show_career_track:


        fig.add_trace(go.Scatter(x=df[df['Номер ']==login]['Начало периода должности'], 
                                y=df[df['Номер ']==login]['Карьерная ступень по Классификатору ФОИР'], 
                                mode='lines',
                                name=login,
                                line=dict(color='rgb(67,67,67)', width=2),
                                connectgaps=True,
                                text=login,
                                textposition="bottom center"
                            ))

        
        color_map = {
                    'Федеральные государственные органы, в т.ч. федеральная государственная служба' : '#E32636', # красный
                    'Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ' : '#FFB841', # оранжевый
                    'Органы местного самоуправления (ОМСУ)' : '#EDFF21', # жёлтый
                    'Подведомственные органам власти (государственные/муниципальные) организации' : '#A2ADD0', # серый
                    'Общественные/общественно-политические организации и НКО ' : '#00693E',  # зелёный              
                    'Государственные корпорации и Институты развития' : '#503D33', # коричневый
                    'Коммерческие компании, государственные/муниципальные предприятия' : '#4169E1', # синий
                    }

        colors_d = df[df['Номер ']==login]['Сфера деятельности по Классификатору ФОИР'].map(color_map) # создаём цветовую схему, соответствующую "Сфере деятельности"

        fig.add_trace(go.Scatter(x=df[df['Номер ']==login]['Начало периода должности'], 
                                y=df[df['Номер ']==login]['Карьерная ступень по Классификатору ФОИР'], 
                                mode='markers',
                                marker=dict(color=colors_d, # используем цветовую схему, соответствующую "Сфере деятельности"
                                            size=9, 
                                                line=dict(width=1,
                                                color='DarkSlateGrey')
                                            ),

                                name=None,
                                hovertemplate='%{x} - %{y} уровень <extra></extra>',
                                showlegend = False


            ))
        fig.update_traces(name=login, showlegend = True)

        fig.update_layout(  

                                showlegend=False,
                                legend_orientation = 'v', 
                                height=540,
                                yaxis = dict(
                                    tickmode = 'array',
                                    tickvals = list(range(1, 13, 1)), ),
                                )

    st.plotly_chart(fig)
    
    color_path = 'data/colors/'

    st.markdown(f'''
                {img_to_html(f'{color_path}красный.png')} - Федеральные государственные органы, в т.ч. федеральная государственная служба; 
                \n{img_to_html(f'{color_path}оранжевый.png')} - Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ; 
                \n{img_to_html(f'{color_path}жёлтый.png')} - Органы местного самоуправления (ОМСУ); 
                \n{img_to_html(f'{color_path}серый.png')} - Подведомственные органам власти (государственные/муниципальные) организации;
                \n{img_to_html(f'{color_path}зелёный.png')} - Общественные/общественно-политические организации и НКО;
                \n{img_to_html(f'{color_path}коричневый.png')} - Государственные корпорации и Институты развития;
                \n{img_to_html(f'{color_path}синий.png')} - Коммерческие компании, государственные/муниципальные предприятия;
                ''', unsafe_allow_html=True)
    
        
### Функции по обработке руглярными выражениями 
def search_regexp(dataframe, regex_for_three_gov_fields, matching_three_gov_fields):
    """На вход функции подаётся датафрейм
    \nНа выход отдаётся этот же датафрейм с новой колонкой 'Сфера деятельности по regex' """

    dataframe['Сфера деятельности по regex'] = np.nan

    for i, _ in dataframe.iterrows(): 

        field_finded_list = []

        for field, fields_list_of_regex in regex_for_three_gov_fields.items():

            for regex_index in range(len(list(fields_list_of_regex))):

                finded = re.search(list(fields_list_of_regex)[regex_index], ''.join(dataframe.loc[i, 'Место работы'].lower().split()))
                if finded is not None:
                    field_finded_list.append(field)
                else:
                    field_finded_list.append('не найдено')

            # field_finded_list = list(set(field_finded_list))
        field_finded_list = list(set(field_finded_list))
        field_finded_list.remove('не найдено')
        field_finded_list = ' '.join(field_finded_list)

        dataframe.at[i, 'Сфера деятельности по regex'] = field_finded_list

    # метчим со сферой деятельности:
    dataframe['Сфера деятельности по regex'] = dataframe['Сфера деятельности по regex'].map(matching_three_gov_fields)

    return dataframe
