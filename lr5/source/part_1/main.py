from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler


def DeleteRowsIfTargetIs2(df):
    """
    Удаление строк с классом 2 и разделение на признаки и целевую переменную
    :param df: исходный набор данных в формате dataframe
    :return: df без столбца target и строк со значением target == 2
             target_column без значений == 2
    """
    df = df[df.target != 2]
    df_without_target_column = df.drop(columns=['target'])
    target_column = df["target"]

    return df_without_target_column, target_column


def TeachTreeAndSelectTwoImportantFeatures(training_input_samples, target_values):
    """
    Обучение дерева и выделение 2-х важных признаков
    :param training_input_samples:
    :param target_values:
    """

    forest = RandomForestRegressor(n_estimators=100, random_state=0)
    # обучение
    forest.fit(training_input_samples, target_values)
    # создание нового набора данных важность - признак
    imp_feature = pd.DataFrame({'importanses': forest.feature_importances_,'name': training_input_samples.axes[1]})
    # сортировка набора данных по убыванию признака важности
    imp_feature = imp_feature.sort_values(by='importanses',ascending=False)

    print(f"Наиболее важный признак {imp_feature.name[imp_feature.axes[0][0]]} "
          f"(важность {imp_feature.importanses[imp_feature.axes[0][0]]})")
    print(f"Второй по важности признак {imp_feature.name[imp_feature.axes[0][1]]} "
          f"(важность {imp_feature.importanses[imp_feature.axes[0][1]]})")


def DataScaling(df):
    """
    Масштабирование данных
    :param df: данные которые нужно масштабировать
    :return: отмасштабированные данные
    """
    feature_name = df.axes[1]  # сохранение названий признаков
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    # преобразование массива в набор данных с присвоением признакам имён
    return pd.DataFrame(df, columns=feature_name)


def main():
    """
    Основная функция
    """
    # чтение данных
    main_df = pd.read_csv('wine.csv')
    main_df = main_df.drop(columns=['id'])

    # удаление строк со значением target == 2
    df_without_target_column, target_column = DeleteRowsIfTargetIs2(main_df)

    # немасштабированные данные
    print("Немасштабированные признаки")
    TeachTreeAndSelectTwoImportantFeatures(df_without_target_column, target_column)

    # масштабирование
    scaled_df_without_target_column = DataScaling(df_without_target_column)
    print("\nМасштабированные признаки")
    TeachTreeAndSelectTwoImportantFeatures(scaled_df_without_target_column, target_column)


# Запуск программы
main()
