import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = None  # Убирает псевдо предупреждения при вызове fillna

def division(df):
    """
    Разделение на исходные данные и целевые
    :param df:
    :return:
    """
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    Y = df['Survived']
    return X, Y


def prepare(df):
    """
    Подготовка данных для обработки
    :param df:
    :return:
    """

    # пропуски только в Age и Fare, оба - числовые - замена на среднее
    df.loc[:, 'Age'].fillna(df['Age'].mean(), inplace=True)
    df.loc[:, 'Fare'].fillna(df['Fare'].mean(), inplace=True)

    # Замена строкового пола на числовой male = 1, female = 0
    new_male_and_female_values = df['Sex'].map({'male': 1, 'female': 0})
    df.isetitem( df.columns.get_loc('Sex'), new_male_and_female_values)

    # преобразование во флаговые признаки порта посадки (one-hot кодирование, оказывается можно и так)
    df = pd.get_dummies(df, columns=['Embarked'])

    # масштабирование данных
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)

    return df


def Regression(x_train, x_test, y_train, y_test, model, search_koef=False, start=0, stop=0, step=0.1):
    """
    обучение и определение качества моделей
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param model:
    :param search_koef: флаг необходимости подбора параметра alpha
    :param start: начало диапазона поиска
    :param stop: конец диапазона поиска
    :param step: шаг поиска
    :return:
    """

    if search_koef:
        grid_searcher = GridSearchCV(model, param_grid={'alpha': np.linspace(start, stop, step)}, cv=5).fit(x_train,
                                                                                                            y_train)
        predict = grid_searcher.predict(x_test)
    else:
        model.fit(x_train, y_train)  # обучение
        predict = model.predict(x_test)  # получение предсказания принадлежности к классу 1

    return mean_squared_error(predict, y_test)  # возврат средней квадратичной ошибки (чем меньше, тем лучше)


def main():
    """
    Основная функция
    :return:
    """

    data = pd.read_csv('./tested.csv')  # основной набор
    X, Y = division(data)  # деление на исходные и целевые
    # b=X.isnull().sum() # определение наличия пропусков в признаках (визуально, поэтому закомментировано)

    X = prepare(X)  # подготовка данных

    # разбиение на обучающие и тестовые данные
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)  # тестовых - 30%

    # определение качества моделей
    qalityLogist = Regression(X_train, X_test, Y_train, Y_test, model=LogisticRegression())  # ошибка логистич регрессии

    qalityRidge = Regression(X_train, X_test, Y_train, Y_test, model=Ridge(), search_koef=True, start=1, stop=500,
                             step=10)  # ошибка обычной регрессии

    qalityLasso = Regression(X_train, X_test, Y_train, Y_test, model=Lasso(2.65e-5))  # ошибка лассо регрессии

    print('Средняя квадратичная ошибка для Логистической регрессии: ', qalityLogist)
    print('Средняя квадратичная ошибка для Ridge регрессии: ', qalityRidge)
    print('Средняя квадратичная ошибка для Лассо регрессии: ', qalityLasso)

    return 0


main()  # запуск программы
