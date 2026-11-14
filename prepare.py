import json, random, requests

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_110m_admin_0_countries.geojson")
geo = requests.get(URL, timeout=60).json()
geo["features"] = [f for f in geo["features"]
                   if f["properties"].get("CONTINENT") == "Africa"]
json.dump(geo, open("africa.geojson", "w"), ensure_ascii=False)

years = list(range(2000, 2021))
data = {}
for f in geo["features"]:
    iso = f["properties"]["ADM0_A3"]
    base = random.uniform(0, 400)
    data[iso] = [round(max(0, base + random.uniform(-30, 30) - 8 * i), 1)
                 for i in range(len(years))]
json.dump({"disease": "Демо-хвороба", "unit": "випадків на 100 тис.",
           "years": years, "max": max(max(v) for v in data.values()),
           "data": data}, open("data.json", "w"), allow_nan=False)
print("Готово:", len(geo["features"]), "країн")