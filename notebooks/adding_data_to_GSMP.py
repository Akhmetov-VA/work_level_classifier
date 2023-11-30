import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from dadata import Dadata
from DaData_token import dadata_token
from OfData_token import ofdata_token
import requests



dadata = Dadata(dadata_token)


# чел вводит место работы:
place_of_work = st.text_area('Введите место работы:', 'Интеллектуальные технологии') 

# если оно не пустое:
if place_of_work is not None:
    # подключаемся к DaData и ищем данные о компании:
    fns_data = dadata.suggest("party", place_of_work)

    # компаний ведь может быть несколько!
    number_of_companies = len(fns_data)

    # чтобы челу сориентироваться - дадим ему название и адрес регистрации компании:
    company_names = [fns_data[i]['value'] for i in range(number_of_companies)] # [fns_data[i]['data']['name']['full_with_opf'] for i in range(number_of_companies)]
    company_adresses = [fns_data[i]['data']['address']['unrestricted_value'] for i in range(number_of_companies)]

    company_names_and_adresses = []
    for i in range(len(company_names)):
        company_names_and_adresses.append(company_names[i] + ' (' + company_adresses[i] + ')')

else:
        st.write('Введите ваше место работы')

# окей, составили подходящий список компаний-адресов - даём челу выбрать нужное:
company_name = st.multiselect('Выберете компанию, подходящую под ваше название', company_names_and_adresses)

# находим в данных нужный ИНН, согласно выбранному названию компании:
if company_name is not None:
    index = company_names_and_adresses.index(company_name[0])
    company_inn = fns_data[index]['data']['inn']

    # РАБОТАЕМ С ИНН через ofdata:
    with st.form('...'):

        try:

            all_data_about_company = requests.get(f'https://api.ofdata.ru/v2/company?key={ofdata_token}&inn={company_inn}')

            # получаем все-все данные о компании:
            all_data_about_company = all_data_about_company.json()

        except:
            pass

        # и среди всех-всех данных отбираем только нужные для обогащения:
        if st.form_submit_button("Найти данные"):

            additional_data = {'Номер': [],
                'ИНН': [],
                'Вид экономической деятельности, ОКВЭД': [],
                'Доп вид экономической деятельности_1': [],
                'Доп вид экономической деятельности_2': [],
                'Доп вид экономической деятельности_3': [],
                'Уставный капитал, тип': [],
                'Уставный капитал, сумма': [],
                'Тип по ОКОГУ': [],
                'Среднесписочная численность сотрудников': [],
                'Категория из реестра СМП': [],
                'Сумма уплаченных налогов за 2020': [],       
                }


            try:
                inn = all_data_about_company['data']['ИНН']

                try:
                    ust_cap_type = all_data_about_company['data']['УстКап']['Тип']
                except:
                    ust_cap_type = None

                try:
                    ust_cap_sum = all_data_about_company['data']['УстКап']['Сумма']
                except:
                    ust_cap_sum = None

                try:
                    okved = all_data_about_company['data']['ОКВЭД']['Наим']
                except:
                    okved = None

                # for n in range(3):
                #     try:
                #         exec(f"okved_dop_{n+1} = {all_data_about_company['data']['ОКВЭДДоп'][n]['Наим']}")            
                #     except:
                #         exec(f"okved_dop_{n+1} = {None}")
                try:
                    okved_dop_1 = all_data_about_company['data']['ОКВЭДДоп'][0]['Наим']
                except:
                    okved_dop_1 = None

                try:
                    okved_dop_2 = all_data_about_company['data']['ОКВЭДДоп'][1]['Наим']
                except:
                    okved_dop_2 = None

                try:
                    okved_dop_3 = all_data_about_company['data']['ОКВЭДДоп'][2]['Наим']
                except:
                    okved_dop_3 = None



                try:
                    schr = all_data_about_company['data']['СЧР']
                except:
                    schr = None


                try:
                    rmsp_kat = all_data_about_company['data']['РМСП']['Кат']
                except:
                    rmsp_kat = None

                try:
                    nalog_sum = all_data_about_company['data']['Налоги']['СумУпл']
                except:
                    nalog_sum = None

                try:
                    okogy = all_data_about_company['data']['ОКОГУ']['Наим']
                except:
                    okogy = None


                
                additional_data['ИНН'].append(inn)
                additional_data['Вид экономической деятельности, ОКВЭД'].append(okved)
                additional_data['Доп вид экономической деятельности_1'].append(okved_dop_1)
                additional_data['Доп вид экономической деятельности_2'].append(okved_dop_2)
                additional_data['Доп вид экономической деятельности_3'].append(okved_dop_3)
                additional_data['Уставный капитал, тип'].append(ust_cap_type)
                additional_data['Уставный капитал, сумма'].append(ust_cap_sum)
                additional_data['Тип по ОКОГУ'].append(okogy)
                additional_data['Среднесписочная численность сотрудников'].append(schr)
                additional_data['Категория из реестра СМП'].append(rmsp_kat)
                additional_data['Сумма уплаченных налогов за 2020'].append(nalog_sum)

            except:
                pass

            st.write(additional_data)

            