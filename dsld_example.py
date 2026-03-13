import requests
import json

BASE_URL = "https://api.ods.od.nih.gov/dsld/v9"

# Search for dietary supplement labels
print("Searching the NIH Dietary Supplement Label Database (DSLD)...")
print("Query: 'Magnesium'\n")

resp = requests.get(f"{BASE_URL}/browse-products/", params={
    "method": "by_keyword",
    "q": "Magnesium",
    "pagesize": 40
})
data = resp.json()

total = data.get("total", {}).get("value", "N/A")
print(f"Total matching products: {total}")
print(f"Showing first 10 results:\n")

for i, hit in enumerate(data.get("hits", []), 1):
    source = hit.get("_source", {})
    dsld_id = hit.get("_id", "N/A")
    name = source.get("fullName", "N/A")
    brand = source.get("brandName", "N/A")
    form = source.get("physicalState", {}).get("langualCodeDescription", "N/A")
    upc = source.get("upcSku", "N/A")

    # Serving info
    servings = source.get("servingSizes", [])
    if servings:
        s = servings[0]
        serving_str = f"{s.get('minQuantity', '')} {s.get('unit', '')}"
    else:
        serving_str = "N/A"

    # Net contents
    contents = source.get("netContents", [])
    content_str = contents[0].get("display", "N/A") if contents else "N/A"

    # Date entered
    events = source.get("events", [])
    date_str = events[0].get("date", "N/A") if events else "N/A"

    print(f"--- Product {i} (DSLD ID: {dsld_id}) ---")
    print(f"  Product:    {name}")
    print(f"  Brand:      {brand}")
    print(f"  Form:       {form}")
    print(f"  Serving:    {serving_str}")
    print(f"  Contents:   {content_str}")
    print(f"  UPC:        {upc}")
    print(f"  Date Added: {date_str}")

    # Fetch full label to get ingredient details
    label_resp = requests.get(f"{BASE_URL}/label/{dsld_id}")
    if label_resp.status_code == 200:
        label = label_resp.json()

        # Dietary ingredients from supplement facts
        di = label.get("dietaryIngredients", [])
        if di:
            print(f"  Supplement Facts ({len(di)} ingredients):")
            for ing in di[:6]:
                ing_name = ing.get("ingredientName", "N/A")
                amount = ing.get("amountPerServing", "")
                unit = ing.get("unit", "")
                dv = ing.get("dvPercent", "")
                dv_str = f" ({dv}% DV)" if dv else ""
                print(f"    - {ing_name}: {amount} {unit}{dv_str}")
            if len(di) > 6:
                print(f"    ... and {len(di) - 6} more")

        # Statements (claims / directions)
        statements = label.get("statements", [])
        for stmt in statements:
            if "Suggested" in stmt.get("type", "") or "Direction" in stmt.get("type", ""):
                print(f"  Directions: {stmt.get('notes', 'N/A')[:120]}...")
                break
    print()
