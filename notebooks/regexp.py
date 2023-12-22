import numpy as np
import pandas as pd
import json
import re


# ------------------------------------------------------------------------
# доп.словарик соответствий ключа у словаря регулярок - со сферой деятельности:

matching_three_gov_fields = {'foiv':'Федеральные государственные органы, в т.ч. федеральная государственная служба',
                            'roiv':'Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ',
                            'mestnie':'Органы местного самоуправления (ОМСУ)',

                            'foiv roiv':'Региональные органы государственной власти, в т.ч. гражданская служба субъектов РФ',
                            'mestnie foiv':'Органы местного самоуправления (ОМСУ)',
                            'mestnie foiv roiv':'Органы местного самоуправления (ОМСУ)', # вот тут вопрос!
                            'mestnie roiv':'Органы местного самоуправления (ОМСУ)',
                    }




# ------------------------------------------------------------------------
# открываем словарь с регурярными выражениями:
# структура такая:
# 3 ключа: 'foiv', 'roiv', 'mestnie'
# каждому ключу соответствует значение - словарь из строк регулярных выражений

with open("Регулярки для 3х сфер деятельности.json") as file: 
  regex_for_three_gov_fields = json.loads(file.read()) 



# ------------------------------------------------------------------------
# функция для поиска ФОИВ, РОИВ и местных органов власти с помощью регулярных выражений в таблице:
  
def finding_regex_and_matching_three_gov_fields(dataframe, regex_for_three_gov_fields, matching_three_gov_fields):
    """На вход функции подаётся датафрейм
    \nНа выход отдаётся этот же датафрейм с новой колонкой 'Сфера деятельности по regex' """

    dataframe['Сфера деятельности по regex'] = np.nan

    for i in range(len(dataframe)): 

        field_finded_list = []

        for field, fields_list_of_regex in regex_for_three_gov_fields.items():

            for regex_index in range(len(list(fields_list_of_regex))):

                finded = re.search(list(fields_list_of_regex)[regex_index], ''.join(dataframe.loc[dataframe.index[i], 'Место работы'].lower().split()))
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