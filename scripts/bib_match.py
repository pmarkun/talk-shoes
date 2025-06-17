import pandas as pd
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path


def extract_bib_number(bib):
    if isinstance(bib, dict):
        number = bib.get("number")
        if number is not None:
            return str(number).strip()
    return None


def best_brand(shoes):
    if isinstance(shoes, list) and shoes:
        # choose highest probability label
        best = max(shoes, key=lambda s: s.get("prob", 0))
        return best.get("label")
    return None


def main(dataset_path: str, csv_path: str, output_path: str | None = None):
    df = pd.read_json(dataset_path)

    df["bib_number"] = df.get("bib").apply(extract_bib_number)
    if "brands" in df.columns:
        df["brand"] = df["brands"]
    else:
        df["brand"] = df.get("shoes").apply(best_brand)

    results = pd.read_csv(csv_path)
    results["number"] = results["number"].astype(str).str.strip()

    merged = df.merge(results, left_on="bib_number", right_on="number", how="left")

    matches = merged["rank"].notna().sum()
    unique_matches = merged.loc[merged["rank"].notna(), "number"].nunique()

    if output_path:
        merged.to_json(output_path, orient="records", default_handler=str)

    print(f"Total records: {len(df)}")
    print(f"Matches found: {matches}")
    print(f"Unique bibs matched: {unique_matches}")

    top100_numbers = results.sort_values("rank").head(100)["number"].astype(str)
    brand_lookup = (merged.dropna(subset=["bib_number"])\
                        .drop_duplicates("bib_number")
                        .set_index("bib_number")["brand"]
                        .to_dict())
    brand_counts = Counter(brand_lookup.get(n, "unknown") for n in top100_numbers)

    print("\nBrand distribution among top 100 runners:")
    for brand, count in brand_counts.items():
        print(f"{brand}: {count}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Match bib numbers to race results CSV")
    parser.add_argument("dataset", help="Path to processed_dataset.json")
    parser.add_argument("csv", help="CSV file with race results")
    parser.add_argument("--output", help="Optional path to save the merged dataset")
    args = parser.parse_args()
    main(args.dataset, args.csv, args.output)
