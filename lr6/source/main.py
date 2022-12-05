import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
from sklearn import datasets
from sklearn.cluster import DBSCAN

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def TrasformToArrayOfNumbers(df):
    """
    Провести кодирование текстовых классов в числовую форму
    :param df: массивоподобные данные, которые нужно кодировать
    :return: массив чисел
    """
    return LabelEncoder().fit_transform(df)


def ReduceFeature(df, n_components, dimensional_reduction_class):
    """
    Уменьшение количества признаков
    :param df: обрабатываемый набор данных в массивоподобном формате
    :param n_components: Предполагоаемое количество компонентов
    :param dimensional_reduction_class: способ уменьшения признаков (PCA или TSNE)
    :return: преобразованный набор данных состоящий из n_components признаков
    """
    return dimensional_reduction_class(n_components=n_components, random_state=0).fit_transform(df)


def DrawObjects(data_after_pca, data_after_tsne, target):
    """
    Построить график объектов
    :param data_after_pca: набор данных после уменьшения количествова признаков с помощью PCA
    :param data_after_tsne: набор данных после уменьшения количествова признаков с помощью TSNE
    :param target: набор целевых классов
    """
    # кодирование классов числами
    codded_target = TrasformToArrayOfNumbers(target)
    codded_target = codded_target[codded_target < 15]

    data_after_pca = data_after_pca[0:len(codded_target)]
    plt.subplot(2, 1, 1)
    plt.scatter(data_after_pca[:, 0], data_after_pca[:, 1], c=codded_target)
    plt.title(f"Объекты с целевым значением от 0 до 14 (метод PCA)")

    data_after_tsne = data_after_tsne[0:len(codded_target)]
    plt.subplot(2, 1, 2)
    plt.scatter(data_after_tsne[:, 0], data_after_tsne[:, 1], c=codded_target)
    plt.title(f"Объекты с целевым значением от 0 до 14 (метод TSNE)")

    plt.show()


def PartOne():
    """
    Первая часть задачния.
    Уменьшение количествова признаков с помощью PCA и TSNE и отрисовка графиков
    """
    main_df = pd.read_csv('data_Mar_64.txt', header=None)

    # разделение на данные и цель
    data, target = np.array(main_df.iloc[:, 1:]), main_df.iloc[:, 0]

    # уменьшение количествова признаков с помощью PCA
    data_after_pca = ReduceFeature(data, 2, PCA)
    print(f"Координаты объекта с индексом 0 (метод PCA): {data_after_pca[0][0]:9.2f}, {data_after_pca[0][1]:9.2f}")

    # уменьшение количествова признаков с помощью TSNE
    data_after_tsne = ReduceFeature(data, 2, TSNE)
    print(f"Координаты объекта с индексом 0 (метод TSNE): {data_after_tsne[0][0]:9.2f}, {data_after_tsne[0][1]:9.2f}")

    # отрисовка графика объектов после обработки PCA и TSNE
    DrawObjects(data_after_pca, data_after_tsne, target)


class MyKMeans:
    def __init__(self, n_clusters=3, n_iters=100):
        """
        :param n_clusters: число кластеров, на которое будут разбиты данные
        :param n_iters: максимальное число итераций
        """
        self.centers = None
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def fit(self, data):
        """
        Провести кластеризацию данных
        :param data: набор данных для обработки
        :return: максимальное число итераций до сходимости
        """
        # фиксация генератора
        np.random.seed(0)
        # получение случайных ценстров всех кластеров
        self.centers = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0), size=(self.n_clusters, data.shape[1]))

        iteration_number = 0
        for iteration_number in range(self.n_iters):
            # получение индекса ближайшего центра для каждой точки
            data_points = pairwise_distances_argmin(data, self.centers)

            new_centers = np.zeros([3, 2])

            for center in range(len(self.centers)):
                center_x = 0
                center_y = 0
                count = 0
                for p in range(len(data_points)):
                    # если точка не относится к текущему кластеру, то пропускаем
                    if data_points[p] != center:
                        continue

                    # пересчёт центров
                    center_x = center_x + data[p][0]
                    # получение суммы и количесвта для находждения среднего
                    center_y = center_y + data[p][1]
                    count = count + 1

                # находим среднее значение
                center_x = center_x / count
                center_y = center_y / count
                new_centers[center] = [center_x, center_y]

            if np.all(self.centers == new_centers):
                break

            self.centers = new_centers

        return iteration_number

    def predict(self, points):
        """
        Получить расстояния между каждой парой выборок points и self.centers
        :param points: набор точек
        :return: Расстояния между каждой парой выборок points и self.centers
        """
        labels = pairwise_distances_argmin(points, self.centers)
        return labels


def RunKMeansClustering(data, n_iters):
    """
    Выполнить класторизацию с помощью
    "Метода k-средних" (MyKMeans)
    :param data: набор данных для проведения класторизации
    :param n_iters: максимальное число итераций
    :return:
    """
    km = MyKMeans(n_clusters=3, n_iters=n_iters)
    convergence = km.fit(data)
    predict = km.predict(data)

    print(f"\nОтвет для объекта с индексом 1 (MyKMeans) (n_iter={n_iters}): {predict[1]}")
    print(f"Алгоритм сошёлся за {convergence} итераций")

    return predict


def RunDBSCANClustering(data):
    """
    Выполнить класторизацию с помощью
    "Основанной на плотности пространственная кластеризации для приложений с шумами" (DBSCAN)
    :param data: набор данных для проведения класторизации
    """
    dbscan = DBSCAN(eps=0.5)
    predict = dbscan.fit_predict(data)

    print(f"\nОтвет для объекта с индексом 1 (DBSCAN): {predict[1]}")
    print(f"Число кластеров: {len(np.unique(predict)) - 1}")
    print(f"Число объектов, отнесённых к выбросам: {sum(np.array(predict) == -1)}")


def PartTwo():
    """
    Вторая часть задачния.
    Класторизация объектов с помощью двух методов:
        1) "Метод k-средних" (MyKMeans)
        2) "Основанная на плотности пространственная кластеризация для приложений с шумами" (DBSCAN)
    """
    # подготовка данных
    n_samples = 1000
    data = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 3.0, 5.0], random_state=0)[0]

    # кластеризация с помощью "Метода k-средних" (MyKMeans)
    predict5 = RunKMeansClustering(data, 5)
    predict100 = RunKMeansClustering(data, 100)
    print(f"Количество изменившихся ответов: {len(predict5) - sum(np.array(predict5) == np.array(predict100))}")

    # кластеризация с помощью "Основанной на плотности пространственная кластеризации для приложений с шумами" (DBSCAN)
    RunDBSCANClustering(data)


def main():
    """
    Основная функция
    """
    # PSA и TSNE
    PartOne()

    # K_Means и DBSCAN
    PartTwo()


# Запуск программы
main()
