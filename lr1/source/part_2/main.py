import pandas as pd # анализ данных
import numpy as np # математика и работа с массивами
from matplotlib import pyplot as plt # построение графиков

# Получить списки ключей и значений из отсортированного по убыванию словаря
def getKeysAndValues(dataset, with_sort):
    dict_dataset = dict(dataset)
    keys = []
    values = []

    if with_sort:
        dict_dataset = sorted(dict_dataset.items(), key=lambda x: x[1], reverse=True)
        for k, v in dict_dataset:
            keys.append(k)
            values.append(v)
    else:
        for k, v in dict_dataset.items():
            keys.append(k)
            values.append(v)

    return (keys, values)

# Отрисовать первую диаграмму.
# Зависимость числа пассажиров зашедших на борт от порта посадки.
# Зависимость средней стоимости билетов и числа пассажиров от класса.
# Зависимость выручки от класса.
def showDiagrams(main_df):
    # подсчёт числа пассажиров, взошедших на борт в городах
    boarding = pd.value_counts(main_df['Embarked'].values, sort=True)

    for i in range(0, len(boarding.index.values), 1):
        if boarding.index.values[i] == 'S':
            boarding.index.values[i] = 'Саутгемптон'
        else:
            if boarding.index.values[i] == 'C':
                boarding.index.values[i] = 'Шербур'
            else:
                boarding.index.values[i] = 'Квинстаун'

    boarding_keys, boarding_values = getKeysAndValues(boarding, True)  # сортировка по убыванию
    top_boarding = len(boarding_keys)

    plt.subplot(131)
    plt.title('Распределение пассажиров по портам посадки')
    plt.bar(np.arange(top_boarding), boarding_values)
    plt.xticks(np.arange(top_boarding), boarding_keys, rotation=0, fontsize=12)
    plt.ylabel('Количество пассажиров')

    # подсчёт числа пассажиров по классам
    classes = pd.value_counts(main_df['Pclass'].values, sort=False)
    classes_keys, classes_values = getKeysAndValues(classes, False)  # без сортировки

    # подсчёт средней стоимости билетов и общей выручки
    average_fare = {}
    revenue = {}
    for i in classes_keys:
        dfs = main_df[['Fare', 'Pclass']].loc[main_df['Pclass'] == i]
        average_fare[i] = dfs['Fare'].mean(axis=0)
        revenue[i] = dfs['Fare'].sum()
    average_fare_keys, average_fare_values = getKeysAndValues(average_fare, False)  # без сортировки
    revenue_keys, revenue_values = getKeysAndValues(revenue, False)  # без сортировки

    plt.subplot(132)
    plt.grid(True)
    plt.title('Зависимость средней стоимости билета и\n'
              'количества пассажиров от класса')
    plt.xticks(classes_keys, rotation=0, fontsize=12)
    plt.yticks(classes_values+average_fare_values)
    plt.xlabel('Классы', color='gray')
    plt.plot(classes_keys, classes_values, 'g',
             average_fare_keys, average_fare_values, 'b--')
    plt.legend(['Количество пассажиров','Средняя стоимость билетов'], loc=2)

    plt.subplot(133)
    plt.grid(True)
    plt.title('Зависимость выручки от класса')
    plt.xticks(classes_keys, rotation=0, fontsize=12)
    plt.yticks(revenue_values)
    plt.xlabel('Классы', color='gray')
    plt.plot(revenue_keys, revenue_values, 'r-.')
    plt.legend(['Выручка от продажи билетов'], loc=1)
    plt.show()

    return

# Заполнить поля значением Unknown.
# Ячейки, содержащие значение NaN, были заполнены строкой "Unknown" с целью идентифкации факта неизвестности
# значения данного параметра для конкретной записи (пасажира). Замена производилось с помощью метода fillna.
def fillNanFields(main_df):
    main_df = main_df.fillna('Unknown')
    return main_df

# Создать дополнительный признак (буквенного кода палубы).
# Введение нового параметра - буквенного кода палубы, позволит в дальнейшем не производить анализ номера каюты
# для определения палубы, с целью использования её в дальнейшей обработке.
def createAnAdditionalAttribute(main_df):
    main_df = main_df.assign(Deck=main_df.Cabin)

    for i in range(len(main_df.Deck)):
        if main_df.Deck[i] != 'Unknown':
            main_df.loc[i, 'Deck'] = main_df.loc[i, 'Deck'][:1]

    return main_df

# Основная функция
def main():
    # 0 - Читаем датасет
    df = pd.read_csv('./titanic.csv', escapechar='`', low_memory=False)
    # 1 - Рисуем диаграммы зависимостей
    showDiagrams(df)
    # 3 - Заполняем поля значением Unknown
    df = fillNanFields(df)
    # 4 - Создаём дополнительный признак
    df = createAnAdditionalAttribute(df)
    # 5 - Сохраняем изменнеия в новый файл
    df.to_csv('./titanic_modded.csv')

# Запускаем программу
main()