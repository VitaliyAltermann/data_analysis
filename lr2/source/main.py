import pandas as pd  # анализ данных


# Найти наиболее распространённое значение в датасете.
# Принимает набор данных и название столбца для определения его самого частого значения.
# возвращает самое частое значение
def FindMostCommonValues(data, column_name):
    return pd.value_counts(data[column_name].values, sort=True).axes[0][0]


# Задание 1:
#           Какая причина выбора школы была самой частой?
#           В качестве ответа приведите соответствующее значение признака.
def FindMostCommonReasonForChoosingSchool(main_df):
    reason_values = {
        "home": "близко к дому",
        "reputation": "репутация школы",
        "course": "предпочтение некоторым предметам",
        "other": "предпочтение некоторым предметам",
    }

    # определение самой частой причины выбора школы
    reason = FindMostCommonValues(main_df, 'reason')

    print("""
Задание 1: Какая причина выбора школы была самой частой?
           В качестве ответа приведите соответствующее значение признака.
Результат: """, reason_values[reason])


# Задание 2:
#           Найдите количество студентов, у родителей которых нет никакого образования.
def FindNumberOfStudentsWhoseParentsHaveNoEducation(main_df):
    no_education_value = 0
    data = main_df[(main_df['Medu'] == no_education_value) & (main_df['Fedu'] == no_education_value)]

    print("""
Задание 2: Найдите количество студентов, у родителей которых нет никакого образования.
Результат: """, len(data))


# Задание 3:
#           Найдите минимальный возраст учащегося школы Mousinho da Silveira.
def FindMinimumAgeOfStudentAtMousinhoDaSilveiraSchool(main_df):
    school_name = 'MS'
    data = main_df[(main_df['school'] == school_name)]

    print("""
Задание 3: Найдите минимальный возраст учащегося школы Mousinho da Silveira.
Результат: """, min(data['age']))


# Задание 4:
#           Найдите количество студентов, имеющих нечетное число пропусков.
def FindNumberOfStudentsWhoHaveAnOddNumberOfAbsences(main_df):
    data = main_df[main_df['absences'] % 2 != 0]

    print("""
Задание 4: Найдите количество студентов, имеющих нечетное число пропусков.
Результат: """, len(data))


# Задание 5:
#           Найдите разность между средними итоговыми оценками студентов, состоящих и не состоящих в
#           романтических отношениях. В качестве ответа приведите число,
#           округленное до двух значащих цифр после запятой.
def FindDifferenceBetweenAverageFinalGradesOfStudentsInAndOutOfRomanticRelationships(main_df):
    data = main_df.groupby('romantic').describe()
    # "{:.2f}".format - округление до 2-х символов после запятой
    result_difference = "{:.2f}".format(data['G3', 'mean']['yes'] - data['G3', 'mean']['no'])

    print("""
Задание 5: Найдите разность между средними итоговыми оценками студентов, состоящих и не состоящих в
           романтических отношениях. В качестве ответа приведите число, 
           округленное до двух значащих цифр после запятой.
Результат: """, result_difference)


# Задание 6:
#           Сколько занятий пропустило большинство студентов с самым частым значением
#           наличия внеклассных активностей?
def HowManyClassesDidMostStudentsWithMostFrequentValueOfHavingExtracurricularActivitiesMiss(main_df):
    activities_is_exist = FindMostCommonValues(main_df, 'activities')
    data_with_activities = main_df[main_df['activities'] == activities_is_exist]
    number_of_students_by_absences = pd.value_counts(data_with_activities['absences'].values, sort=True)

    print("""
Задание 6: Сколько занятий пропустило большинство студентов с самым частым значением
           наличия внеклассных активностей?
Результат: """, number_of_students_by_absences.axes[0][0])


# Основная функция
def main():
    # 0 - Читаем датасет
    df = pd.read_csv('./math_students.csv', escapechar='`', low_memory=False)
    # Задание 1
    FindMostCommonReasonForChoosingSchool(df)
    # Задание 2
    FindNumberOfStudentsWhoseParentsHaveNoEducation(df)
    # Задание 3
    FindMinimumAgeOfStudentAtMousinhoDaSilveiraSchool(df)
    # Задание 4
    FindNumberOfStudentsWhoHaveAnOddNumberOfAbsences(df)
    # Задание 5
    FindDifferenceBetweenAverageFinalGradesOfStudentsInAndOutOfRomanticRelationships(df)
    # Задание 6
    HowManyClassesDidMostStudentsWithMostFrequentValueOfHavingExtracurricularActivitiesMiss(df)


# Запускаем программу
main()
