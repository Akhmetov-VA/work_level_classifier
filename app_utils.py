import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import dadata
from dadata import Dadata
import requests

ofdata_token = 'JrFCUBe4XNHhBYCy'
dadata_token = '41070744a4728056044eab22b9adb74beff1bfe9'

dadata = Dadata(dadata_token)

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


        all_data_about_company = requests.get(f'https://api.ofdata.ru/v2/company?key={ofdata_token}&inn={company_inn}')

        # получаем все-все данные о компании:
        all_data_about_company = all_data_about_company.json()

        # st.write(all_data_about_company)

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

        st.write(additional_data)
                
    return additional_data