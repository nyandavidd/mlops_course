import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_report():
    print("Загрузка данных...")
    df = pd.read_csv("data/all_v2.csv")

   
    
    # Сортируем по времени
    df = df.sort_values(by="year")

    # Имитация: 
    # Reference - старые данные (до 2022 года)
    # Current - новые данные (с 2022 года)
    # Мы хотим проверить, сильно ли изменились данные с тех пор
    reference_data = df[df["year"] < 2021]
    current_data = df[df["year"] >= 2021]

    print(f"Reference size: {len(reference_data)}")
    print(f"Current size: {len(current_data)}")

    # Выбираем колонки для анализа (цена и признаки)
    # Убираем дату, так как она очевидно меняется
    columns_to_analyze = ["price", "area", "kitchen_area", "rooms", "level", "levels"]
    
    reference_data = reference_data[columns_to_analyze]
    current_data = current_data[columns_to_analyze]

    print("Генерация отчета Evidently...")
    # Создаем отчет о дрейфе данных
    report = Report(metrics=[
        DataDriftPreset(), 
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Сохраняем в HTML
    report.save_html("drift_report.html")
    print("Отчет сохранен как 'drift_report.html'")

if __name__ == "__main__":
    generate_report()