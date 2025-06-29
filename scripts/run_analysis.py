import json

corrida = json.load(open("output/bibs_data_progressive.json", "r"))

list_of_runners = {}

for runner in corrida:
    bib = runner.get("bib", {})
    if isinstance(bib, list):
        bib = bib[0]  # Assuming we want the first entry if multiple exist
    try:
        number = int(bib.get("number"))
    except (ValueError, TypeError):
        continue
    category = bib.get("category")
    label = [l.get("label") for l in runner.get("shoes")]
    if number is None or category is None:
        continue
    if number in list_of_runners:
        list_of_runners[number]["count"] += 1
        list_of_runners[number]["categories"].append(category)
        list_of_runners[number]["labels"].extend(label)
    else:
        list_of_runners[number] = {
            "count": 1,
            "categories": [category],
            "labels": label
        }

print(list_of_runners)
print("Total runners:", len(corrida))
print("Total unique runners:", len(list_of_runners))


#Pick up the most common category and label for each runner
for number, data in list_of_runners.items():
    data["most_common_category"] = max(set(data["categories"]), key=data["categories"].count)
    data["most_common_label"] = max(set(data["labels"]), key=data["labels"].count)

# Create a dict to store category -> label distribution
category_label_distribution = {}

# Iterate through all runners
for number, data in list_of_runners.items():
    category = data["most_common_category"]
    label = data["most_common_label"]
    
    # Initialize category if not exists
    if category not in category_label_distribution:
        category_label_distribution[category] = {}
    
    # Initialize or increment label count in this category
    if label not in category_label_distribution[category]:
        category_label_distribution[category][label] = 0
    category_label_distribution[category][label] += 1

# Convert counts to percentages
for category, labels in category_label_distribution.items():
    total_in_category = sum(labels.values())
    for label in labels:
        labels[label] = round((labels[label] / total_in_category) * 100, 2)

print("\nCategory to Label Distribution (%):")
print(json.dumps(category_label_distribution, indent=2))
