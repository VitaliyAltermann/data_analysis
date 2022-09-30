import random  # генератор случайных чисел
import pandas as pd  # анализ данных
import numpy as np  # математика и работа с массивами
from PIL import Image  # работа с изображениями


# 1 - Вывести заданное количество случайных чисел
def showImage(df):
    result_mass = np.zeros((28, 28), dtype=np.uint8)

    index = random.randint(0, 4200)

    for i in range(0, 784, 1):
        result_mass[i // 28, i % 28] = 255 - df[f"pixel{i}"].values[index]

    Image.fromarray(result_mass).show()

def main():
    # 0 - Читаем датасет
    df = pd.read_csv('./numbers.csv', escapechar='`', low_memory=False)

    # 1 - Выводим заданное количество случайных чисел
    for i in range(0, int(input("Введите желаемое число выводимых изображений: ")), 1):
        showImage(df)

main()
