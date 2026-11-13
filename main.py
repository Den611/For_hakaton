import json
import random

def generate_forecast_data(filename="data.json"):
    years = list(range(2000, 2031))
    iso_codes = [
        "TZA", "SAH", "COD", "SOM", "KEN", "SDN", "TCD", "ZAF", "LSO", 
        "ZWE", "BWA", "NAM", "SEN", "MLI", "MRT", "BEN", "NER", "NGA", 
        "CMR", "TGO", "GHA", "CIV", "GIN", "GNB", "LBR", "SLE", "BFA", 
        "CAF", "COG", "GAB", "GNQ", "ZMB", "MWI", "MOZ", "SWZ", "AGO", 
        "BDI", "MDG", "GMB", "TUN", "DZA", "ERI", "MAR", "EGY", "LBY", 
        "ETH", "DJI", "SOL", "UGA", "RWA", "SDS"
    ]
    
    data_dict = {}
    error_dict = {}
    max_val = 0.0
    
    random.seed(42)
    for iso in iso_codes:
        values = []
        errors = []
        current_val = random.uniform(50, 300)
        
        for year in years:
            if year <= 2025:
                current_val += random.uniform(-15, 15)
                current_val = max(10, current_val)
                values.append(round(current_val, 1))
                errors.append(0.0)
            else:
                # Прогноз і розрахунок похибки засобами Python
                current_val += random.uniform(-8, 12)
                current_val = max(10, current_val)
                years_ahead = year - 2025
                error_margin = round(current_val * (0.04 * years_ahead), 1)
                
                values.append(round(current_val, 1))
                errors.append(error_margin)
                
        data_dict[iso] = values
        error_dict[iso] = errors
        if max(values) > max_val:
            max_val = max(values)

    payload = {
        "disease": "Демо-хвороба (Історія та Прогноз з 2026)",
        "unit": "випадків на 100 тис.",
        "years": years,
        "max": max_val,
        "data": data_dict,
        "errors": error_dict
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
    print(f"Дані успішно згенеровано у {filename}")

if __name__ == "__main__":
    generate_forecast_data()