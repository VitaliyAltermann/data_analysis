import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def ShowGraphOfMinimalErrorDependenceOnTrait(thresholds_and_min_errors):
    """
    Показать график зависимости минимальной ошибки от признака
    :param thresholds_and_min_errors:
    :return:
    """
    #
    attributes = []
    min_errors = []
    for i in range(0, len(thresholds_and_min_errors)):
        attributes.append(thresholds_and_min_errors[i][0])
        min_errors.append(thresholds_and_min_errors[i][2])

    plt.plot(attributes, min_errors)
    plt.title("Зависимость минимального значения критерия ошибки от признака")
    plt.xlabel("Признак")
    plt.ylabel("Минимальное значение ошибки")
    plt.show()


def ShowPartitioningGraphForOptimalTraitByThreshold(main_df, target_attribute_name, optimal_attribute, optimal_threshold, optimal_error):
    """
    Отрисовать график разбиения для оптимального признака по порогу
    :param main_df: главный датасет
    :param target_attribute_name: название целевого признака
    :param optimal_attribute: оптимальный признак
    :param optimal_threshold: оптимальный порог
    :param optimal_error: значение ошибки
    :return:
    """
    plt.scatter(main_df[optimal_attribute], main_df[target_attribute_name])
    plt.axvline(x=optimal_threshold, color="red")
    plt.xlabel(optimal_attribute)
    plt.ylabel(target_attribute_name)
    plt.title(f"Визуализация разбиения для оптимального признака {optimal_attribute} "
              f"по порогу {optimal_threshold} (при значении критерия ошибки {optimal_error})")
    plt.show()


def AwarenessCriterion(R_df, target_attribute_name):
    """
    Критерий информативности, с помощью которого можно оценить качество распределения
    целевой переменной среди объектов множества R.
    Определение энтропии.
    :param R_df: множество в формате dataframe
    :param target_attribute_name: название целевого признака
    :return:
    """
    targen_value = R_df[target_attribute_name]
    mean = np.mean(targen_value)

    # по всем строкам (их индексам)
    sum = 0
    for i in R_df.axes[0]:
        sum = sum + ((targen_value[i] - mean) ** 2)
    h = 1 / len(targen_value) * sum

    return h


def SplitNode(node, attribute_name, threshold):
    """
    Разбиение узла
    :param node: узел (значения) в формате dataframe
    :param attribute_name: название признака по которому будет проходить разбиение
    :param threshold: порог ("<=" - лево, ">" - право)
    :return:
    """
    attribute = node[attribute_name]
    rl = node[attribute <= threshold]
    rr = node[attribute > threshold]
    return rl, rr


def QError(node, attribute_name, threshold, target_attribute_name):
    """
    Вычисление ошибки
    :param node: узел (значения) в формате dataframe
    :param attribute_name: название признака по которому будет проходить разбиение
    :param threshold: порог
    :param target_attribute_name: название целевого признака
    :return:
    """
    # Разбиваем узел
    rl, rr = SplitNode(node, attribute_name, threshold)

    # Вычисляем ошибки
    if len(rl) == 0:
        return (len(rr) / len(node)) * AwarenessCriterion(rr, target_attribute_name)
    elif len(rr) == 0:
        return (len(rl) / len(node)) * AwarenessCriterion(rl, target_attribute_name)
    return (len(rl) / len(node)) * AwarenessCriterion(rl, target_attribute_name) + \
           (len(rr) / len(node)) * AwarenessCriterion(rr, target_attribute_name)


def ErrorDependencyAndOptimalSplit(node, attribute_name, target_attribute_name):
    """
    Определение завистимости ошибки от порогового значения для признака
    :param node: узел (значения) в формате dataframe
    :param attribute_name: название признака по которому будет проходить разбиение
    :param target_attribute_name: название целевого признака
    :return: лучшее пороговое значение и минимальная ошибка
    """
    # Определим уникальные значения порогов определение ошибки
    unique_thresholds = np.unique(node[attribute_name])

    # Расчитаем значения ошибки для всех порогов определение ошибки
    q_array = []
    for threshold in unique_thresholds:
        q_array.append(QError(node, attribute_name, threshold, target_attribute_name))

    # Определим оптимальный порог и минимальную ошибку
    optimal_t = 0
    min_error = min(q_array)
    for i in range(0, len(q_array)):
        if q_array[i] == min_error:
            optimal_t = unique_thresholds[i]
            break

    # Построение графика зависимости ошибки от порогового значения (закомментировано для быстроты работы)
    plt.plot(unique_thresholds,q_array)
    plt.title(f"Зависимость значения критерия ошибки от порогового значения для признака {attribute_name}")
    plt.xlabel("Пороговое значение")
    plt.ylabel("Значение ошибки")
    plt.show()
    return optimal_t, min_error


def FindOfThresholdsAndMinErrors(df, target_attribute_name):
    """
    Нахождение оптимальных порогов
    :param df: набор данных в формате dataframe
    :param target_attribute_name: имя целевого признака
    :return: Оптимальный признак,
             Оптимальный порог,
             Значение ошибки.
    """
    # Пройдём по всем признакам и получим пороговые значения и ошибки
    attribute_and_threshold_and_min_errors = []
    for attribute_name in df.columns:
        if attribute_name != target_attribute_name:
            threshold, min_err = ErrorDependencyAndOptimalSplit(df, attribute_name, target_attribute_name)
            attribute_and_threshold_and_min_errors.append((attribute_name, threshold, min_err))
    return attribute_and_threshold_and_min_errors


def main():
    """
    Основная функция
    """
    # Загружаем датасет
    data = pd.read_csv('boston.csv')
    target_attribute_name = "MEDV"

    # Находим оптимальные пороги
    thresholds_and_min_errors = FindOfThresholdsAndMinErrors(data, target_attribute_name)

    # Cортируем по минимальной ошибке, чтобы получить лучший результат.
    # В итоге получем оптимальный признак, оптимальный порог и значение ошибки.
    optimal_attribute, optimal_threshold, optimal_error = sorted(thresholds_and_min_errors, key=lambda x: x[2])[0]

    # Рисуем график зависимости минимальной ошибки от признака
    ShowGraphOfMinimalErrorDependenceOnTrait(thresholds_and_min_errors)

    # Рисуем график разбиения для оптимального признака по порогу
    ShowPartitioningGraphForOptimalTraitByThreshold(data,
                                                    target_attribute_name,
                                                    optimal_attribute,
                                                    optimal_threshold,
                                                    optimal_error)


# Запуск программы
main()
