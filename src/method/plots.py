import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np



def fitness_plot(fitness_history, *, show=True):
    avg_fitness_per_generation = [sum(scores)/len(scores) for scores in fitness_history]
    plt.plot(avg_fitness_per_generation)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Evolution of Fitness Over Generations")
    if show:
        plt.show()


def services_plot(df_before, df, res_id, services, size_factor=100, alpha=0.05, size_factor_2=100, alpha_2=0.05, x=20, y=50, x_2=10, y_2=50):
    df_before = df_before.copy()
    df = df.copy()
    services = services.copy()
    df['new_service'] = df.index.map(lambda x: "новый сервис" if x in res_id.keys() else "нет сервиса")
    df.loc[((df_before['capacity'] != 0) & (df['new_service'] == "новый сервис")), 'new_service'] = "был сервис + новый"
    df.loc[((df_before['capacity'] != 0) & (df['new_service'] == "нет сервиса")), 'new_service'] = "был сервис"
    df['diff_capacity'] = df['capacity'] - df_before['capacity']

    calc_temp = df[df['new_service'] == 'новый сервис'].copy()

    max_capacity = calc_temp['capacity'].max()
    min_capacity = calc_temp['capacity'].min()

    # Устанавливаем диапазон для markersize
    size_factor = size_factor  # множитель для корректного масштаба (можно настроить)

    # Вычисляем размеры точек на основе нормализованных значений capacity
    if max_capacity == min_capacity:
        calc_temp.loc[:, 'marker_size'] = 300  # Задаем фиксированный размер
    else:
        calc_temp.loc[:, 'marker_size'] = ((calc_temp['capacity'] - min_capacity) / (max_capacity - min_capacity) + alpha) * size_factor


    # Теперь рисуем точки с соответствующим размером, в зависимости от capacity
    ax = df.plot(alpha=1, color="#ddd", figsize=[10, 10])

    # Отображаем поликлиники
    services[services['our_service'] == "+"].plot(
        ax=ax,
        markersize=40,
        color="#108a00",
        label='old service',
        marker='*'
    )

    # Отображаем новый сервис с размерами точек в зависимости от capacity
    for idx, row in calc_temp.iterrows():
        size = calc_temp.loc[idx, 'marker_size']  # Получаем размер точки для этой строки
        ax.scatter(row.geometry.centroid.x, row.geometry.centroid.y, color='red', s=size, marker='o')

        # Добавляем подпись с capacity рядом с точкой
        ax.text(row.geometry.centroid.x + 20.8, row.geometry.centroid.y + 50.8,  # Смещение для текста
                f"{calc_temp.loc[idx, 'capacity']:.0f}", fontsize=12,)
                    # bbox=dict(facecolor='white', boxstyle='round,pad=0.1'))

    calc_temp_2 = df[df['new_service'] == 'был сервис + новый'].copy()

    max_capacity = calc_temp_2['diff_capacity'].max()
    min_capacity = calc_temp_2['diff_capacity'].min()

    # Устанавливаем диапазон для markersize
    size_factor = size_factor_2  # множитель для корректного масштаба (можно настроить)

    # Вычисляем размеры точек на основе нормализованных значений capacity
    if max_capacity == min_capacity:
        calc_temp_2.loc[:, 'marker_size'] = size_factor_2  # Задаем фиксированный размер
    else:
        calc_temp_2.loc[:, 'marker_size'] = ((calc_temp_2['diff_capacity'] - min_capacity) / (max_capacity - min_capacity) + alpha_2) * size_factor_2


    for idx, row in calc_temp_2.iterrows():
        
        size = calc_temp_2.loc[idx, 'marker_size'] 
        ax.scatter(row.geometry.centroid.x, row.geometry.centroid.y, color='purple', s=size, marker='^')

        # Добавляем подпись с capacity рядом с точкой
        ax.text(row.geometry.centroid.x - x_2, row.geometry.centroid.y + y_2,  # Смещение для текста
                f"+{df[df['new_service'] == 'был сервис + новый'].loc[idx, 'diff_capacity']:.0f}", fontsize=12,)
        
    legend_handles = [
        Line2D([0], [0], marker='*', color='w', label='old service', markerfacecolor="#108a00", markeredgecolor="#108a00", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='new service', markerfacecolor="red", markeredgecolor="red", markersize=8),
        Line2D([0], [0], marker='^', color='w', label='upgraded service', markerfacecolor="purple", markeredgecolor="purple", markersize=8),
    ]
    ax.legend(handles=legend_handles, title="Service types", loc="upper left")

    ax.set_axis_off()

def connect_blocks_plot(id, matrix, blocks, matrix_best):
    
    rename_dict = {i: id[i] for i in range(len(id))}

    difference_matrix = matrix - matrix_best
    difference_matrix.reset_index(drop=True, inplace=True)
    difference_matrix.columns=difference_matrix.index
    # Гео-данные кварталов
    centroids = blocks.geometry.centroid
    coords = np.array([[point.x, point.y] for point in centroids])

    # Нормализуем значения для цветовой шкалы
    norm = Normalize(vmin=difference_matrix[difference_matrix > 0].min().min(), vmax=(difference_matrix.max().max()-3))
    cmap = plt.cm.coolwarm  # Выбираем цветовую карту

    # Рисуем карту
    fig, ax = plt.subplots(figsize=(10, 10))
    blocks.plot(ax=ax, alpha=1, color="#ddd")

    for i in range(difference_matrix.shape[0]):
        for j in range(i + 1, difference_matrix.shape[0]):
            
            diff = difference_matrix.iloc[i, j]
            if diff != 0:  # Рисуем только, если есть разница
                real_index_i = rename_dict[i]
                real_index_j = rename_dict[j]

                # Используем реальные индексы для получения координат
                x_coords = [coords[real_index_i, 0], coords[real_index_j, 0]]
                y_coords = [coords[real_index_i, 1], coords[real_index_j, 1]]
                ax.plot(
                    x_coords,
                    y_coords,
                    color=cmap(norm(diff)),  # Цвет зависит от значения разницы
                    linewidth=2
                )

    # Добавляем цветовую шкалу
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.04)
    cbar.set_label("Time difference (min)")

    # for idx, (x, y) in enumerate(coords):
    #     ax.text(x, y, str(idx), fontsize=8, ha='center', va='center', )
                # bbox=dict(facecolor='white', edgecolor='black'))

    # Отключаем оси для карты
    ax.set_axis_off()
    plt.show()
