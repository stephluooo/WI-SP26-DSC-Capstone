import requests

BASE_URL = "https://api.fda.gov/food/event.json"

params = {
    "search": 'products.industry_name:"Dietary Conventional Foods"',
    "limit": 10
}

response = requests.get(BASE_URL, params=params)
data = response.json()

print(f"Total matching records: {data['meta']['results']['total']}")
print()

for i, record in enumerate(data["results"], 1):
    print(f"--- Record {i} ---")
    print(f"  Report #:  {record['report_number']}")
    print(f"  Date:      {record.get('date_started', 'N/A')}")
    print(f"  Outcomes:  {record.get('outcomes', [])}")
    print(f"  Reactions: {record.get('reactions', [])}")
    for p in record.get("products", []):
        print(f"  Product:   {p.get('name_brand', 'N/A')} ({p.get('role', '')})")
    consumer = record.get("consumer", {})
    if consumer:
        print(f"  Consumer:  age={consumer.get('age', 'N/A')} {consumer.get('age_unit', '')}, gender={consumer.get('gender', 'N/A')}")
    print()
