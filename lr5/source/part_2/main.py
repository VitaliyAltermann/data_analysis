import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class Bagging:
    def __init__(self, n_estimators=10):
        # Число используемых деревьев
        self.n_estimators = n_estimators

        # Список объектов класса DecisionTreeRegressor, к которым уже был применён метод fit
        # Данный список необходимо заполнить в методе fit и использовать для предсказания в методе predict
        self.regressors = []

    def fit(self, training_input_samples, target_values):
        """
        Обучить
        :param training_input_samples: Входная обучающая выборка
        :param target_values: Целевые значения для входной обучающей выборки
        :return:
        """
        # обучение модели
        for i in range(self.n_estimators):
            # перезагрузка генератора случайных чисел
            np.random.seed(i)
            # получение случайных индексов, на которых будет обучение
            ind = np.random.choice(np.arange(training_input_samples.shape[0]),
                                   size=training_input_samples.shape[0])
            # создание модели
            model = DecisionTreeRegressor()
            # обучение модели (дерева)
            model.fit(training_input_samples.iloc[ind], target_values.iloc[ind])
            # добавление обученного дерева в список
            self.regressors.append(model)

    def predict(self, training_input_samples):
        """
        Спрогнозировать
        :param training_input_samples: Входная обучающая выборка
        :return: Приблизительные значения цели.
        """
        # обнуление предсказаний
        y_pred = np.zeros(training_input_samples.shape[0])
        # для всех обученных деревьев
        for i in range(self.n_estimators):
            # прибавление предсказания очередного дерева
            y_pred += self.regressors[i].predict(training_input_samples)

        # усреднение
        return y_pred / self.n_estimators


def RandomForest(number_of_trees,
                 training_input_samples,
                 target_values,
                 test_training_input_samples,
                 test_target_values):
    """
    Определение ошибки случайного леса
    :param number_of_trees: число деревьев в лесу
    :param training_input_samples: Входная обучающая выборка
    :param target_values: Целевые значения для входной обучающей выборки
    :param test_training_input_samples: Входная тестовая выборка
    :param test_target_values: Целевые значения для входной тестовой выборки
    :return: средняя ошибка
             ошибка out-of-bag
    """
    model = RandomForestRegressor(n_estimators=number_of_trees, random_state=0, oob_score=True)
    model.fit(training_input_samples, target_values)

    # получение средней ошибки
    mse = mean_squared_error(test_target_values, model.predict(test_training_input_samples))
    mse = round(mse / 1000)

    # получение ошибки out-of-bag
    oob = round((1 - model.oob_score_), 2)

    return mse, oob


def Run(count_test_values,
        training_input_samples,
        target_values,
        test_training_input_samples,
        test_target_values):
    """
    Провести сравнительный тест
    :param count_test_values: количество деревьшев
    :param training_input_samples: Входная обучающая выборка
    :param target_values: Целевые значения для входной обучающей выборки
    :param test_training_input_samples: Входная тестовая выборка
    :param test_target_values: Целевые значения для входной тестовой выборки
    """
    # бэггинг
    model = Bagging(count_test_values)
    model.fit(training_input_samples, target_values)
    result = mean_squared_error(test_target_values, model.predict(test_training_input_samples))
    print(f"Ошибка бэггинга с деревьями в количестве {count_test_values} шт: {round(result / 1000)}")

    # случайный лес
    mse, oob = RandomForest(count_test_values,
                            training_input_samples,
                            target_values,
                            test_training_input_samples,
                            test_target_values)
    print(f"Ошибка случайного леса с деревьями в количестве {count_test_values} шт:\n"
          f"Средняя ошибка = {mse}\n"
          f"Ошибка out-of-bag = {oob}")
    print("\n")


def main():
    """
    Основная функция
    """
    data = pd.read_csv('data.csv')

    # разделение на признаки и цель
    features, targets = data.iloc[:, :100], data.iloc[:, 100]
    # разделение на обучающую и тестовую части 6000 - в обучающей
    training_input_samples, target_values = features[:6000], targets[:6000]
    test_training_input_samples, test_target_values = features[6000:], targets[6000:]

    # на 1 дереве
    Run(1,
        training_input_samples,
        target_values,
        test_training_input_samples,
        test_target_values)

    # на 5 деревьях
    Run(5,
        training_input_samples,
        target_values,
        test_training_input_samples,
        test_target_values)

    # на 25 деревьях
    Run(25,
        training_input_samples,
        target_values,
        test_training_input_samples,
        test_target_values)

    # на 100 деревьях
    Run(100,
        training_input_samples,
        target_values,
        test_training_input_samples,
        test_target_values)


# Запуск программы
main()
