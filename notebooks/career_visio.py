import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import msoffcrypto
import base64
from pathlib import Path


#---------------------------------------------------------------------------------------------
# ОТОБРАЖАЕМ ИЗОБРАЖЕНИЯ В MARKDOWN


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html


#---------------------------------------------------------------------------------------------
# ОТКРЫВАЕМ ТАБЛИЦУ, ГДЕ ЕСТЬ 3 КОЛОНКИ:
# 'Начало периода должности'
# 'Сфера деятельности'
# 'Уровень должности (УД max 13)'
# 
# ИЛИ ПОДТЯГИВАЕМ ТАБЛИЦУ С ПРЕДИКТОМ

passwd = 'class2023'

decrypted_workbook = io.BytesIO()
with open('новые данные ВШГУ - 3 части - 01.12.23/Базы для IT_карьерные траектории.xlsx', 'rb') as file:
    office_file = msoffcrypto.OfficeFile(file)
    office_file.load_key(password=passwd)
    office_file.decrypt(decrypted_workbook)

df = pd.read_excel(decrypted_workbook, sheet_name='пробный', converters={"Уровень должности (УД max 13)":int, "Начало периода должности": int })

logins_to_show_career_track = st.multiselect('Выберете логин, чью карьерную траекторию отображать:', df['Логин'].sort_values().unique())


# color_map = {
#                 'Федеральные государственные органы, в т.ч. федеральная государственная служба' : 'red',
#                 'Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ' : 'orange',
#                 'Органы местного самоуправления (ОМСУ)' : 'yellow',
#                 'Подведомственные органам власти (государственные/муниципальные) организации' : 'grey',
#                 'Общественные/общественно-политические организации и НКО ' : 'green',                
#                 'Государственные корпорации и Институты развития' : 'brown',
#                 'Коммерческие компании, государственные/муниципальные предприятия' : 'blue',
#                 }

# colors_d = df['Сфера деятельности'].map(color_map)

if logins_to_show_career_track == None:
    pass
else:
    fig = go.Figure()

    for login in logins_to_show_career_track:


        fig.add_trace(go.Scatter(x=df[df['Логин']==login]['Начало периода должности'], 
                                y=df[df['Логин']==login]['Уровень должности (УД max 13)'], 
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

        colors_d = df[df['Логин']==login]['Сфера деятельности'].map(color_map) # создаём цветовую схему, соответствующую "Сфере деятельности"

        fig.add_trace(go.Scatter(x=df[df['Логин']==login]['Начало периода должности'], 
                                y=df[df['Логин']==login]['Уровень должности (УД max 13)'], 
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