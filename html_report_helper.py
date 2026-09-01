import numpy as np
import pandas as pd
try:
    from IPython.display import HTML, display
except ImportError:  # IPython нужен только для интерактивного показа; в проде хватает return_html
    def HTML(x):
        return x

    def display(*args, **kwargs):
        return None


def display_semaphore(colour, width=46, height=18, return_html=False):
    """
    Функция с кодами цветов светофоров (для таблицы с критериями выставления светофоров).
    """

    red_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red {
                    background-color: #ca1d1d;
                }
                .amber {
                    background-color: #ffd600;
                }
                .green {
                    background-color: #04d930;
                }
                .gray {
                    background-color: #ccc;
                }
                
            </style>
        </head>
        <body>
        <div class="traffic-light">
                <div class="light red"></div>
                <div class="light gray"></div>
                <div class="light gray"></div>
            </div>
        </body>
        """
    yellow_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red {
                    background-color: #ca1d1d;
                }
                .amber {
                    background-color: #ffd600;
                }
                .green {
                    background-color: #04d930;
                }
                .gray {
                    background-color: #ccc;
                }
                
            </style>
        </head>
        <body>
        <div class="traffic-light">
                <div class="light gray"></div>
                <div class="light amber"></div>
                <div class="light gray"></div>
            </div>
        </body>
        """
    green_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red {
                    background-color: #ca1d1d;
                }
                .amber {
                    background-color: #ffd600;
                }
                .green {
                    background-color: #04d930;
                }
                .gray {
                    background-color: #ccc;
                }
                
            </style>
        </head>
        <body>
        <div class="traffic-light">
                <div class="light gray"></div>
                <div class="light gray"></div>
                <div class="light green"></div>
            </div>
        </body>
        """
    grey_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red {
                    background-color: #ca1d1d;
                }
                .amber {
                    background-color: #ffd600;
                }
                .green {
                    background-color: #04d930;
                }
                .gray {
                    background-color: #ccc;
                }
                
            </style>
        </head>
        <body>
        <div class="traffic-light">
                <div class="light gray"></div>
                <div class="light gray"></div>
                <div class="light gray"></div>
            </div>
        </body>
        """

    if colour == "green":
        if not return_html:
            display(HTML(green_light_html))
        else:
            return green_light_html
    elif colour == "yellow" or colour == "amber":
        if not return_html:
            display(HTML(yellow_light_html))
        else:
            return yellow_light_html
    elif colour == "red":
        if not return_html:
            display(HTML(red_light_html))
        else:
            return red_light_html
    elif colour == "grey" or colour == "gray":
        if not return_html:
            display(HTML(grey_light_html))
        else:
            return grey_light_html


def show_criteria_semaphore(
    green_criterion, yellow_criterion, red_criterion, gray_criterion, styles
):
    """
    Функция формирования таблицы с критериями выставления светофора по тесту.

    Parameters:

    green_criterion : str
        Критерий на зеленый светофор.

    yellow_criterion : str
        Критерий на желтый светофор.

    red_criterion : str
        Критерий на красный светофор

    styles : list
        Список с селекторами заголовков, ячеек и границ таблицы, которые задают форматирование.

    Return:

    _ : pd.DataFrame
        Табличка переданного форматирования.

    """

    criterion = pd.DataFrame(columns=["comment", "colour"])
    criterion["colour"] = ["green", "yellow", "red", "grey"]

    criterion.loc[criterion.colour == "green", "comment"] = green_criterion
    criterion.loc[criterion.colour == "yellow", "comment"] = yellow_criterion
    criterion.loc[criterion.colour == "red", "comment"] = red_criterion
    criterion.loc[criterion.colour == "grey", "comment"] = gray_criterion

    criterion.loc[criterion["colour"] == "green", "colour_img"] = display_semaphore(
        "green", return_html=True
    )
    criterion.loc[criterion["colour"] == "yellow", "colour_img"] = display_semaphore(
        "yellow", return_html=True
    )
    criterion.loc[criterion["colour"] == "red", "colour_img"] = display_semaphore(
        "red", return_html=True
    )
    criterion.loc[criterion["colour"] == "grey", "colour_img"] = display_semaphore(
        "grey", return_html=True
    )

    criterion.rename(
        {"colour_img": "Результат", "comment": "Критерий"}, axis=1, inplace=True
    )
    criterion.drop("colour", axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


def show_criteria_semaphore_without_red(green_criterion, yellow_criterion, styles):
    """
    Функция формирования таблицы с критериями выставления светофора по тесту.

    Parameters:

    green_criterion : str
        Критерий на зеленый светофор.

    yellow_criterion : str
        Критерий на желтый светофор.

    styles : list
        Список с селекторами заголовков, ячеек и границ таблицы, которые задают форматирование.

    Return:

    _ : pd.DataFrame
        Табличка переданного форматирования.

    """

    criterion = pd.DataFrame(columns=["comment", "colour"])
    criterion["colour"] = ["green", "yellow"]

    criterion.loc[criterion.colour == "green", "comment"] = green_criterion
    criterion.loc[criterion.colour == "yellow", "comment"] = yellow_criterion

    criterion.loc[criterion["colour"] == "green", "colour_img"] = display_semaphore(
        "green", return_html=True
    )
    criterion.loc[criterion["colour"] == "yellow", "colour_img"] = display_semaphore(
        "yellow", return_html=True
    )

    criterion.rename(
        {"colour_img": "Результат", "comment": "Критерий"}, axis=1, inplace=True
    )
    criterion.drop("colour", axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


def show_criteria_semaphore_without_yellow(green_criterion, red_criterion, styles):
    """
    Функция формирования таблицы с критериями выставления светофора по тесту.

    Parameters:

    green_criterion : str
        Критерий на зеленый светофор.

    red_criterion : str
        Критерий на красный светофор

    styles : list
        Список с селекторами заголовков, ячеек и границ таблицы, которые задают форматирование.

    Return:

    _ : pd.DataFrame
        Табличка переданного форматирования.

    """

    criterion = pd.DataFrame(columns=["comment", "colour"])
    criterion["colour"] = ["green", "red"]

    criterion.loc[criterion.colour == "green", "comment"] = green_criterion
    criterion.loc[criterion.colour == "red", "comment"] = red_criterion

    criterion.loc[criterion["colour"] == "green", "colour_img"] = display_semaphore(
        "green", return_html=True
    )
    criterion.loc[criterion["colour"] == "red", "colour_img"] = display_semaphore(
        "red", return_html=True
    )

    criterion.rename(
        {"colour_img": "Результат", "comment": "Критерий"}, axis=1, inplace=True
    )
    criterion.drop("colour", axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


def show_df(
    df, green_threshold={}, red_threshold={}, type_threshold="abs", precision=2
):
    """
    Функция для отображения датафрейма с возможностью добавления колонок со светофорами.

    Псевдокод:
        1. определяем метрики для выставления светофоров исходя из ключей переданных порогов светофров;
        2. определяем, какие величины (абсолютные или относительные) используются для выставления светофоров;
        3. для каждой метрики в зависимости от направления её оптимизации ставим светофоры;
        4. добавляем столбец светофоров сразу за столбцом метрики в итоговой таблице.

    Parameters:

        df: pd.DataFrame

        green_threshold: dict, float
            Условие для выставления зелёного светофора, указаное в процентах от изменения метрики, или
            в абсолютных значениях.

        red_threshold: dict, float
            Условие для выставления красного светофора, указаное в процентах от изменения метрики, или
            в абсолютных значениях.

        type_threshold: str, default='abs'
            Задаёт тип значений переданных в условиях выставления светофоров.

                * 'abs': абсолютные значения;
                * 'relative': процент от изменения метрики.

        precision : int
            Точность, до которой возвращается значение метрики в отчете в таблице с метриками.
    """

    df = df.copy()
    # проверка пересечения наименований метрик
    metrics = [*(set(green_threshold.keys()) & set(red_threshold.keys()))]

    # row_colours = []
    if metrics:
        if type_threshold == "relative":
            compare_base = 100 * (1 - df[metrics] / df[metrics].iloc[0])
        elif type_threshold == "abs":
            compare_base = df[metrics]

        colours_by_metric = []
        for metric in metrics:
            # для проверки длины порогов оберну их в np.array
            green, red = (
                np.array(green_threshold[metric]),
                np.array(red_threshold[metric]),
            )
            # значений порогов должно быть либо по количеству горизонтов, либо одно
            assert (len(df) == green.size) | (green.size == 1), (
                "Некорректные по длине границы светофоров."
            )
            assert (len(df) == red.size) | (red.size == 1), (
                "Некорректные по длине границы светофоров."
            )

            mvals = compare_base[metric]
            if (green <= red).all():
                condlist = np.array(
                    [
                        mvals >= red,
                        (mvals > green) & (mvals < red),
                        mvals <= green,
                        pd.isna(mvals),
                    ]
                )
                colours = np.select(condlist, ["red", "yellow", "green", "grey"])
            elif (green > red).all():
                condlist = np.array(
                    [
                        mvals >= green,
                        (mvals > red) & (mvals < green),
                        mvals <= red,
                        pd.isna(mvals),
                    ]
                )
                colours = np.select(condlist, ["green", "yellow", "red", "grey"])
            else:
                raise ValueError("Некорректно выставлены границы светофоров.")

            colours_by_metric.append(colours)
            # порядковый номер колонки с метрикой
            ncol = int(np.where(df.columns == metric)[0][0])
            if type_threshold == "relative":
                df.insert(ncol + 1, metric + "Прирост, п.п.", mvals)
                ncol = ncol + 1
            df.insert(ncol + 1, metric + "<br>semaphore", colours)

        colours_by_metric = np.array(colours_by_metric).T

    for column in df.columns:
        if "semaphore" in str(column):
            df["Результат"] = np.array(
                [
                    display_semaphore(colour, return_html=True)
                    for colour in df[column].values
                ]
            )
    df.drop(columns=[metrics[0] + "<br>semaphore"], inplace=True)

    return df, colours_by_metric
