import pickle  # или joblib, в зависимости от того, как сохранена модель
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from joblib import load

app = Flask(__name__)

# Загрузка модели
model = load("model40.joblib")
print('loaded')
print(model)

inflation_dct = {
    (2018, 1): 1,
    (2018, 2): 1.0031,
    (2018, 3): 1.00520651,
    (2018, 4): 1.008121608879,
    (2018, 5): 1.01195247099274,
    (2018, 6): 1.0157978903825124,
    (2018, 7): 1.0207753000453867,
    (2018, 8): 1.0235313933555092,
    (2018, 9): 1.0236337464948448,
    (2018, 10): 1.0252715604892366,
    (2018, 11): 1.028860010950949,
    (2018, 12): 1.0340043110057036,
    (2019, 1): 1.0426899472181514,
    (2019, 2): 1.0532211156850548,
    (2019, 3): 1.057855288594069,
    (2019, 4): 1.0612404255175703,
    (2019, 5): 1.0643180227515712,
    (2019, 6): 1.0679367040289265,
    (2019, 7): 1.068363878710538,
    (2019, 8): 1.0705006064679592,
    (2019, 9): 1.0679314050124362,
    (2019, 10): 1.0662227147644163,
    (2019, 11): 1.06760880429361,
    (2019, 12): 1.070598108945632,
    (2020, 1): 1.0744522621378363,
    (2020, 2): 1.0787500711863875,
    (2020, 3): 1.0823099464213026,
    (2020, 4): 1.0882626511266198,
    (2020, 5): 1.0972952311309707,
    (2020, 6): 1.1002579282550242,
    (2020, 7): 1.1026784956971853,
    (2020, 8): 1.1065378704321256,
    (2020, 9): 1.1060952552839527,
    (2020, 10): 1.105320988605254,
    (2020, 11): 1.1100738688562566,
    (2020, 12): 1.117955393325136,
    (2021, 1): 1.1272344230897346,
    (2021, 2): 1.1347868937244359,
    (2021, 3): 1.1436382314954865,
    (2021, 4): 1.1511862438233567,
    (2021, 5): 1.157863124037532,
    (2021, 6): 1.1664313111554099,
    (2021, 7): 1.1744796872023822,
    (2021, 8): 1.1781205742327097,
    (2021, 9): 1.1801233792089052,
    (2021, 10): 1.1872041194841587,
    (2021, 11): 1.200382085210433,
    (2021, 12): 1.2119057532284532,
    (2022, 1): 1.2348433804049265,
    (2022, 2): 1.248433804049265,
    (2022, 3): 1.343433804049265,
    (2022, 4): 1.3644,
    (2022, 5): 1.366,
    (2022, 6): 1.3612,
    (2022, 7): 1.3559,
    (2022, 8): 1.3488,
    (2022, 9): 1.3495,
    (2022, 10): 1.3519,
    (2022, 11): 1.3569,
    (2022, 12): 1.3675,
    (2023, 1): 1.379,
    (2023, 2): 1.3853,
    (2023, 3): 1.3904,
    (2023, 4): 1.3957,
    (2023, 5): 1.4,
    (2023, 6): 1.4052,
    (2023, 7): 1.4141,
    (2023, 8): 1.4181,
    (2023, 9): 1.4304,
    (2023, 10): 1.4423,
    (2023, 11): 1.4583,
    (2023, 12): 1.4689,
    (2024, 1): 1.4815,
    (2024, 2): 1.4916,
    (2024, 3): 1.4974,
    (2024, 4): 1.5049,
    (2024, 5): 1.516,
    (2024, 6): 1.5257,
    (2024, 7): 1.5431,
    (2024, 8): 1.5462,
    (2024, 9): 1.5536,
    (2024, 10): 1.5653,
    (2024, 11): 1.5877,
    (2024, 12): 1.6087,
}





# Загрузка данных для тепловой карты
def load_heatmap_data():
    try:
        print("\n=== Статистика данных для тепловой карты ===")
        print("Начинаем загрузку данных...")
        heatmap_data = pd.read_csv('second_verse.csv')
        print(f"Всего строк в CSV: {len(heatmap_data):,}")
        print("Столбцы в данных:", heatmap_data.columns.tolist())
        
        required_columns = ['price', 'geo_lat', 'geo_lon', 'region', 'area']
        if not all(col in heatmap_data.columns for col in required_columns):
            raise ValueError(f"Отсутствуют необходимые столбцы. Доступные столбцы: {heatmap_data.columns.tolist()}")
        
        heatmap_data = heatmap_data[heatmap_data['region'] == 2661]
        print(f"\nПосле фильтрации по региону СПб: {len(heatmap_data):,} объектов")
        
        heatmap_data = heatmap_data[(heatmap_data['price'] >= 1_000_000) & (heatmap_data['price'] <= 50_000_000)]
        print("Отфильтровано строк по цене:", len(heatmap_data))
        
        heatmap_data = heatmap_data[heatmap_data['area'] > 0]
        print("Отфильтровано строк с положительной площадью:", len(heatmap_data))
        
        if len(heatmap_data) > 30000:
            print(f"\nИсходное количество объектов ({len(heatmap_data):,}) превышает лимит в 30,000")
            heatmap_data = heatmap_data.sample(n=30000, random_state=42)
        print(f"\nИтоговое количество объектов для тепловой карты: {len(heatmap_data):,}")
        
        heatmap_data['price_per_sqm'] = heatmap_data['price'] / heatmap_data['area']
        
        print("\nДиапазоны значений:")
        print(f"Цены за кв.м: от {heatmap_data['price_per_sqm'].min():,.0f} до {heatmap_data['price_per_sqm'].max():,.0f} рублей/кв.м")
        print(f"Широты: от {heatmap_data['geo_lat'].min():.6f} до {heatmap_data['geo_lat'].max():.6f}")
        print(f"Долготы: от {heatmap_data['geo_lon'].min():.6f} до {heatmap_data['geo_lon'].max():.6f}")
        
        heatmap_points = heatmap_data[['price_per_sqm', 'geo_lat', 'geo_lon', 'year']].values.tolist()
        print("\n=== Конец статистики ===\n")
        return heatmap_points
    except Exception as e:
        print("Ошибка при загрузке данных:", str(e))
        return []

# Загружаем данные при запуске приложения
heatmap_points = load_heatmap_data()

@app.route("/get_heatmap_data", methods=["GET"])
def get_heatmap_data():
    year = request.args.get('year', type=int)
    if year is None:
        return jsonify({'error': 'Year parameter is required'}), 400
    
    # Если запрошен 2025 год, используем данные за 2020
    if year == 2025:
        year = 2020
    
    filtered_points = [point for point in heatmap_points if point[3] == year]
    return jsonify({'points': filtered_points})

@app.route("/", methods=["GET", "POST"])
def index():
    print('Метод запроса:', request.method)
    if request.method == "POST":
        print('Получены POST данные:', request.form)
        try:
            data = request.form.to_dict()
            prediction = predict_price(data)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'prediction': prediction})
            
            return render_template("index.html", prediction=prediction, heatmap_points=heatmap_points)
        except Exception as e:
            print('Ошибка при обработке запроса:', str(e))
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': str(e)}), 400
            return render_template("index.html", error=str(e), heatmap_points=heatmap_points)
            
    return render_template("index.html", heatmap_points=heatmap_points)


def predict_price(data):
    print('Начало predict_price')
    print('Полученные данные:', data)

    try:
        if 'year' not in data or not data['year']:
            raise ValueError("Год не указан")
        
        year = int(data["year"])
        if year < 2018 or year > 2025:
            raise ValueError(f"Год должен быть между 2018 и 2025, получено: {year}")
            
        print(f'Год успешно преобразован: {year}')
        
        month = int(data["month"])
        geo_lat = float(data["geo_lat"])
        geo_lon = float(data["geo_lon"])
        building_type = data["building_type"]
        object_type = data["object_type"]
        levels = int(data["levels"])
        level = int(data["floor"])
        rooms = int(data["rooms"])
        area = float(data["area"])
        kitchen_area = float(data["kitchen_area"])
        print('Все данные успешно преобразованы')
        
        features = np.array(
            [
                [
                    geo_lat,
                    geo_lon,
                    building_type,
                    level,
                    levels,
                    rooms,
                    area,
                    kitchen_area,
                    object_type,
                    year,
                    month,
                ]
            ]
        )
        print('Массив признаков создан:', features)
        
        price = model.predict(features)
        print('Предсказание выполнено:', price)
        
        # Применяем инфляцию для 2025 года
        if year == 2025:
            # Используем коэффициент инфляции за соответствующий месяц 2024 года
            inflation_key = (2024, month)
            inflation_coefficient = inflation_dct.get(inflation_key, 1.0)
            price = price * inflation_coefficient
            print(f'Применена инфляция за {month} месяц 2024 года: {inflation_coefficient}')
        
        return round(price[0], 2)
        
    except ValueError as e:
        print('Ошибка при обработке данных:', str(e))
        raise
    except Exception as e:
        print('Неожиданная ошибка:', str(e))
        raise


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
