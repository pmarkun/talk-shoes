import pandas as pd
import numpy as np

def ci_wilson(k, n, z=1.96):
    """Wilson score interval para contagem k em n"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = (z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0, centre - margin), min(1, centre + margin)

def brand_ci(df, group_cols=[], N=None, z=1.96):
    """
    Calcula proporção + IC95% (Wilson) por marca.
    - group_cols=[]  → geral
    - group_cols=['gender'] → por gênero
    """
    res = []

    # ---------- contagem k ----------
    keys = group_cols + ["brands"]
    grouped = df.groupby(keys).size().rename('k')

    # ---------- total n ----------
    if group_cols:
        totals = df.groupby(group_cols).size().rename('n')
        stats = (
            grouped.reset_index()
            .merge(totals.reset_index(), on=group_cols)
        )
    else:
        n_total = len(df)
        stats = grouped.reset_index().assign(n=n_total)

    # ---------- loop linhas ----------
    for _, row in stats.iterrows():
        k, n = row['k'], row['n']
        p_hat = k / n

        low, high = ci_wilson(k, n, z)

        if N:                                 # correção de população finita
            fpc = np.sqrt((N - n) / (N - 1))
            se  = np.sqrt(p_hat*(1-p_hat)/n) * fpc
            low, high = max(0, p_hat - z*se), min(1, p_hat + z*se)

        entry = {c: row[c] for c in group_cols} if group_cols else {'group':'all'}
        entry.update({
            'brand': row["brands"],
            'n': int(n),
            'k': int(k),
            'prop': round(p_hat*100, 2),
            'ci_low': round(low*100, 2),
            'ci_high': round(high*100, 2),
            'moe' : round((high - low) * 100, 2)
        })  # type: ignore
        res.append(entry)

    return pd.DataFrame(res)



def load_dataset(file_path):
    """
    Load the pd dataset from a JSON file.
    """
    df = pd.read_json(file_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    return df

def clean_dataset(df):
    """
        Remove invalid rows from the dataset.
    """
    df = df[df['demographic'].notna() & df['demographic'].apply(lambda x: isinstance(x, dict) and 'age' in x)]
    
    df = df[df['shoes'].notna() & df['shoes'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    print(f"Filtered dataset to {len(df)} rows with valid demographic and shoes data.")
    return df
    
def clean_gender(df, probability_threshold=0.9):
    
    df = df[df['demographic'].apply(lambda x: x['gender']['prob'] >= probability_threshold)]
    print(f"Filtered dataset to {len(df)} rows with gender probability > {probability_threshold}.")
    return df

def clean_age(df, allowed_ages=["20-29", "30-39", "40-49", "50-59", "60-69"]):
    """
    Remove rows where the age is less than the specified threshold.
    """
    df = df[df['demographic'].apply(lambda x: x['age']['label'] in allowed_ages)]
    print(f"Filtered dataset to {len(df)} rows with ages in {'/'.join(allowed_ages)}.")
    return df

def clean_shoes(df, probability_threshold=0.9):
    """
    Remove rows where the shoes list is empty or where no shoe has a probability >= probability_threshold.
    """        
    # Check if shoes is a list and has at least one element
    df = df[df['shoes'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    # Filter entries where ANY shoe has a probability >= probability_threshold
    df = df[df['shoes'].apply(lambda x: any(shoe['prob'][0] >= probability_threshold for shoe in x))]

    # Create a new column with the definitive label of the shoe
    def get_definitive_label(shoes):
        if len(shoes) == 1:
            return shoes[0]['label'][0]
        elif len(shoes) == 2:
            if shoes[0]['label'][0] == shoes[1]['label'][0]:
                return shoes[0]['label'][0]
            else:
                return max(shoes, key=lambda x: x['prob'][0])['label'][0]
        else:
            # More than 2 shoes, check for duplicates
            labels = [shoe['label'][0] for shoe in shoes]
            counts = {label: labels.count(label) for label in set(labels)}
            most_common_label = max(counts, key=counts.get) # type: ignore
            # If there's a tie, return the one with the highest probability
            max_prob = -1
            for shoe in shoes:
                if shoe['label'][0] == most_common_label and shoe['prob'][0] > max_prob:
                    max_prob = shoe['prob'][0]
                    most_common_label = shoe['label'][0]
            return most_common_label
    
    df['brands'] = df['shoes'].apply(get_definitive_label)
    return df

def clean_results(df):
    df["gender"] = df["demographic"].apply(lambda x: x["gender"]["label"])
    df["age"] = df["demographic"].apply(lambda x: x["age"]["label"])
    df.drop(columns=["demographic", "shoes"], inplace=True)
    return df

def brand_distribution(df, decimals: int = 2) -> pd.DataFrame:
    """
    Calcula a distribuição percentual das marcas (brands) em três escopos:
      • geral (todas as linhas)
      • por gender
      • por age

    Retorna um DataFrame long com colunas:
      scope   -> 'overall', 'gender', 'age'
      group   -> 'all' para o geral, ou valor de gender / age
      brand   -> nome da marca
      percent -> % arredondada com `decimals` casas
    """
    frames = []

    # ----------------- 1. Geral -----------------
    overall = (df['brands']
               .value_counts(normalize=True)
               .mul(100)
               .round(decimals)
               .rename_axis('brand')
               .reset_index(name='percent'))
    overall['scope'] = 'overall'
    overall['group'] = 'all'
    frames.append(overall[['scope', 'group', 'brand', 'percent']])

    # ----------------- 2. Por gender -----------------
    gender = (df.groupby(['gender', 'brands'])
                .size()
                .div(df.groupby('gender').size(), level='gender')  # divide pelo total daquele gender
                .mul(100)
                .round(decimals)
                .rename('percent')
                .reset_index())                 # ← sem colisão de nomes
    gender['scope'] = 'gender'
    gender = gender.rename(columns={'gender': 'group'})
    frames.append(gender[['scope', 'group', 'brands', 'percent']]
                  .rename(columns={'brands': 'brand'}))

    # ----------------- 3. Por age -----------------
    age = (df.groupby(['age', 'brands'])
              .size()
              .div(df.groupby('age').size(), level='age')
              .mul(100)
              .round(decimals)
              .rename('percent')
              .reset_index())
    age['scope'] = 'age'
    age = age.rename(columns={'age': 'group'})
    frames.append(age[['scope', 'group', 'brands', 'percent']]
                 .rename(columns={'brands': 'brand'}))
    
    # ----------------- 4. Por folder -----------------
    folder = (df.groupby(['folder', 'brands'])
                .size()
                .div(df.groupby('folder').size(), level='folder')
                .mul(100)
                .round(decimals)
                .rename('percent')
                .reset_index())
    folder['scope'] = 'folder'
    folder = folder.rename(columns={'folder': 'group'})
    frames.append(folder[['scope', 'group', 'brands', 'percent']]
                  .rename(columns={'brands': 'brand'}))

    # Empilha tudo
    result = pd.concat(frames, ignore_index=True)
    return result

import json

def processAnnotations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        annotations = []
        for item in data:
            for ann in item["annotations"]:
                for r in ann["result"]:
                    if r["from_name"] == "brand" and r["value"]["choices"][0] not in ["Desconhecido","Outra"]:
                        annotations.append({
                            "brand": r["value"]["choices"][0],
                            "folder": item["data"].get("folder",""),
                            "image": item["data"]["image"],
                        })

    # Criar um DataFrame a partir das anotações
    df = pd.DataFrame(annotations)

    # Calcular a distribuição de marcas
    brand_distribution = round(df["brand"].value_counts(normalize=True) * 100,2)
    # renomeia para percent
    brand_distribution = brand_distribution.rename_axis('brand').reset_index(name='percent')
    brand_distribution['scope'] = 'z_annotations'
    brand_distribution['group'] = 'all'
    brand_distribution = brand_distribution[['scope', 'group', 'brand', 'percent']]

    return df, brand_distribution

def correct_proportions(o_series: pd.Series,
                        C: np.ndarray,
                        brands_order: list[str]) -> pd.Series:
    """Aplica NNLS para corrigir vetor observado (Series em %) usando matriz C."""
    from scipy.optimize import nnls

    vec = o_series.reindex(brands_order, fill_value=0) / 100
    vec = vec.values
    p, _ = nnls(C, vec)
    p = p / p.sum()
    return pd.Series(p * 100, index=brands_order)


from argparse import ArgumentParser

if __name__ == "__main__":
    # Load the dataset
    parser = ArgumentParser(description="Process the dataset.")
    parser.add_argument("file", type=str, default="prova2004-dataset.json",
                        help="Path to the dataset file.")
    parser.add_argument("--annotated", type=str,
                        help="Path to the annotated dataset file.")  
    parser.add_argument("--output", type=str,
                        help="Path to the output Excel file.")
    parser.add_argument("--participants", type=int, default=0,
                        help="Number of participants. If 0, uses the length of the dataset.")
    parser.add_argument("--detailed", action="store_true",
                        help="Generate detailed output.")
    parser.add_argument("--confusion", type=str,
                    help="Path to confusion_prob.csv para correção de erro de classificação.")

    args = parser.parse_args()
    file_path = args.file
    output_file = args.output
    annotated = args.annotated

    df_raw = load_dataset(file_path)
    df_raw = clean_dataset(df_raw)
    df = df_raw.copy()

    
    N = args.participants | len(df)
    PROBABILITY=0.85
    
    df = clean_gender(df, probability_threshold=PROBABILITY)
    df = clean_age(df)
    df = clean_shoes(df, probability_threshold=PROBABILITY)
    df = clean_results(df)
    dist = brand_distribution(df)
    
    if annotated:
        df_annotated, distribution_annotated = processAnnotations(annotated)
        dist = pd.concat([dist, distribution_annotated], ignore_index=True)
   
    pivot = dist.pivot_table(index=['scope', 'group'],
                            columns='brand',
                            values='percent',
                            fill_value=0)

    overall_ci = brand_ci(df, group_cols=[], N=N)
    prova_ci = brand_ci(df, group_cols=["folder"], N=N)
    gender_ci = brand_ci(df, group_cols=["gender"], N=N)
    age_ci = brand_ci(df, group_cols=["age"], N=N)
    overall_corr = pd.DataFrame()
    
    print("Distribuição geral:")
    print(pivot)
    # dentro do main, após calcular overall_ci
    if args.confusion:
        C = pd.read_csv(args.confusion, index_col=0).values
        brands_order = list(pd.read_csv(args.confusion, index_col=0).columns)

        overall_corr = correct_proportions(overall_ci.set_index('brand')['prop'],
                                        C,
                                        brands_order).reset_index()
        overall_corr.columns = ['brand', 'prop_corrected']
        print("\n*** Geral corrigido pela matriz de confusão ***")
        print(overall_corr)

    
    if args.detailed:
        print("\nDistribuição geral com IC:")
        print(overall_ci)

        print("\nDistribuição por folder com IC:")
        print(prova_ci)
        print("\nDistribuição por gênero com IC:")
        print(gender_ci)
        print("\nDistribuição por idade com IC:")
        print(age_ci)

    # ------------------------------------------------------------
    # Depois de calcular pivot, overall_ci, prova_ci, gender_ci, age_ci
    # ------------------------------------------------------------
    if output_file:
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            # 1. Distribuição percentual (pivot “wide”)
            pivot.reset_index().to_excel(writer,
                                        sheet_name="Distribuição %",
                                        index=False)

            # 2. Intervalo de confiança - Geral
            overall_ci.to_excel(writer,
                                sheet_name="Geral IC",
                                index=False)

            if args.detailed:
                # 3. IC por folder (pasta de fotos)
                prova_ci.to_excel(writer,
                                sheet_name="Folder IC",
                                index=False)

                # 4. IC por gênero
                gender_ci.to_excel(writer,
                                sheet_name="Gênero IC",
                                index=False)

                # 5. IC por faixa etária
                age_ci.to_excel(writer,
                                sheet_name="Idade IC",
                                index=False)
                
                # 6. Ajustado pela matriz de confusão
            if args.confusion:
                overall_corr.to_excel(writer,
                                    sheet_name="Geral Corrigido",
                                    index=False)

        print(f"Arquivo salvo em: {output_file}")
