import pandas as pd  # анализ данных
import numpy as np  # математика и работа с массивами
from matplotlib import pyplot as plt  # построение графиков
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def PrepareDataSet(main_df):
    """
    Подготовка датасета для дальнейшего использования
    :param main_df: основной датасет
    :return: main_df: модифицированный основной датасет
             training_df: тренеровочный набор
             test_df: тестовый набор
    """
    # удаление времени высадки
    main_df = main_df.drop('dropoff_datetime', axis=1)
    # преобразование даты и времени
    main_df.pickup_datetime = pd.to_datetime(main_df.pickup_datetime)
    # создание нового признака - дня недели (номер) dt - для восприятия как datetime
    main_df['day_of_week'] = main_df.pickup_datetime.dt.day_of_week
    main_df['day_of_year'] = main_df.pickup_datetime.dt.day_of_year
    # сортировка по хронологии
    main_df = main_df.sort_values(by='pickup_datetime')
    # преобразование длительности поездки в логарифмический вид для удобства определения зависимости
    main_df['log_trip_duration'] = np.log1p(main_df.trip_duration)

    # Разбиение на тестовую и тренировочную выборки
    training_df = main_df[:10 ** 6]
    test_df = main_df[10 ** 6:]

    return main_df, training_df, test_df


def PlotNumberOfTripsAsFunctionOfDayOfWeek(df):
    """
    Задание 1:
              Постройте график соответствующий количеству поездок в зависимости
              от дня недели по обучающей выборке.
    :param df: датасет для построения графика
    """
    trips = pd.value_counts(df['day_of_week'].values, sort=False)

    plt.title('Распределение поездок по дням недели')
    plt.bar(np.arange(len(trips)), trips)
    plt.xticks([i for i in range(7)], ('пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс'))
    plt.show()


def AddBinaryVariableToAttributes(main_df):
    """
    Задание 2:
              Добавьте к признакам бинарную переменную, которая равна 1 для двух аномальных дней
              и 0 во все остальные дни. Для этого понадобиться модифицировать функцию create_features.
    :param main_df: основной датасет
    :return: модифицированный объект main_df
    """
    days_and_trips = pd.value_counts(main_df['day_of_year'].values, sort=True)
    beck_of_days_and_trips = days_and_trips.axes[0][len(days_and_trips) - 1]
    pre_beck_of_days_and_trips = days_and_trips.axes[0][len(days_and_trips) - 2]

    main_df = main_df.assign(is_anomaly_day=0)

    for i in range(len(main_df.is_anomaly_day)):
        item = main_df.day_of_year[i]
        if (item == beck_of_days_and_trips) or (item == pre_beck_of_days_and_trips):
            main_df.loc[i, 'is_anomaly_day'] = 1

    return main_df


def AddDayOfWeekAsFeatureForLearning(df):
    """
    Задание 3.1:
                Добавьте день недели в качестве признака для обучения.
    :param df: исходный набор данных
    :return: x_df, y_df
    """
    x_df = pd.concat([df.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),
                      df.pickup_datetime.apply(lambda x: x.hour),
                      df.day_of_week],
                     axis=1,
                     keys=['day', 'hour', 'day_of_week'])
    y_df = df.log_trip_duration
    return x_df, y_df


def DoOneHotCoding(train_df, test_df):
    """
    Задание 3.2:
                Заново проведите one-hot кодирование.
    :param train_df: обучающий датасет
    :param test_df: тестовый датасет
    :returns train_df, test_df: модифицированные обучающий и тестовый датасеты
    """
    ohe = ColumnTransformer([("One hot", OneHotEncoder(sparse=False), [1, 2])], remainder="passthrough")
    # получено 32 признака - 24 флаговых - час дня, 7 флаговых - день недели, 1 числовой - день года
    train_df = ohe.fit_transform(train_df)
    test_df = ohe.transform(test_df)
    return train_df, test_df


def ScaleSingleRealTrait(list_of_data):
    """
    Задание 4.1:
                Отмасштабируйте единственный вещественный признак.
    :param list_of_data:
    :return: модифицированный объект list_of_data
    """
    list_mean = np.average(list_of_data, axis=0)[len(list_of_data[0]) - 1]
    for i in range(len(list_of_data)):
        list_of_data[i][len(list_of_data[0]) - 1] = list_of_data[i][len(list_of_data[0]) - 1] / list_mean
    return list_of_data


def TeachLassoRegression(x_training_df, y_training_df, x_test_df, y_test_df):
    """
    Задание 4.2:
                Обучите на полученных данных Lasso регрессию, в качества параметра alpha возьмите 2.65e-05.
    :param x_training_df: обучающий датасет
    :param y_training_df: обучающий датасет
    :param x_test: тестовый датасет
    :param y_test: тестовый датасет
    """
    # модель - Лассо-регрессия параметр - из задания
    model = Lasso(alpha=2.65e-05)

    # обучение модели
    print("Начало обучения Lasso регресси")
    model.fit(x_training_df, y_training_df)
    print("Lasso регресся обучена")

    # возврашение средней квадратичной ошибки
    print("После обучения Lasso регресси получили следующую среднеквадратичную ошибки:",
          mean_squared_error(model.predict(x_test_df), y_test_df))


def main():
    """
    Основная функция
    """

    # 0 - Читаем датасет
    main_df, training_df, test_df = PrepareDataSet(pd.read_csv('./train.zip',
                                                               compression='zip',
                                                               header=0,
                                                               sep=',',
                                                               quotechar='"'))
    # Задание 1
    PlotNumberOfTripsAsFunctionOfDayOfWeek(training_df)
    # Задание 2
    main_df = AddBinaryVariableToAttributes(main_df)
    # main_df.to_csv('./train_modded.csv')
    # Задание 3.1
    x_training_df, y_training_df = AddDayOfWeekAsFeatureForLearning(training_df)
    x_test_df, y_test_df = AddDayOfWeekAsFeatureForLearning(test_df)
    # Задание 3.2
    x_training_df, x_test_df = DoOneHotCoding(x_training_df, x_test_df)
    # Задание 4.1
    x_training_df = ScaleSingleRealTrait(x_training_df)
    x_test_df = ScaleSingleRealTrait(x_test_df)
    # Задание 4.2
    TeachLassoRegression(x_training_df, y_training_df, x_test_df, y_test_df)


# Запускаем программу
main()
