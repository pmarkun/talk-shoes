"""
Script para enriquecer dados de atletas de um arquivo JSON com informações de um arquivo CSV com o resultado da corrida.

Este script realiza as seguintes etapas:
1. Carrega um arquivo JSON original contendo dados de atletas.
2. Transforma o JSON em uma lista de objetos, onde cada objeto representa um atleta.
3. Carrega um arquivo CSV contendo informações dos resultados das corridas. As colunas são: N do Peito,Nome do Atleta,Posicao,Tempo,Genero,Local,KM,Categoria.
4. Enriquece os objetos JSON com os dados correspondentes do CSV, usando o "N do Peito" (bib number) como chave.
5. Gera um novo arquivo JSON ("resultado_combinado.json") com os dados enriquecidos.
6. Gera um arquivo CSV ("comparacao_genero.csv") para comparar o gênero informado no JSON original
   com o gênero informado no CSV de entrada, juntamente com outras informações relevantes do CSV.

Configuração:
- Os nomes dos arquivos de entrada/saída e os nomes das colunas relevantes do CSV
  podem ser configurados no início do script.

Uso:
- Coloque os arquivos JSON_INPUT_FILE e CSV_INPUT_FILE no mesmo diretório do script.
- Execute o script: python seu_script.py
- Os arquivos JSON_OUTPUT_FILE e CSV_COMPARISON_FILE serão gerados no mesmo diretório.
"""

import json
import csv
import re
import os

# --- Configurações ---
JSON_INPUT_FILE = "modelo.json"
CSV_INPUT_FILE = "resultado_corrida.csv"
JSON_OUTPUT_FILE = "resultado_combinado.json"
CSV_COMPARISON_FILE = "comparacao_genero.csv"

# Nomes das colunas relevantes no CSV de entrada
CSV_BIB_COLUMN = "N do Peito"
CSV_RANK_COLUMN = "Posicao"
CSV_GENDER_COLUMN = "Genero"
CSV_DISTANCE_COLUMN = "KM"
CSV_LOCAL_COLUMN = "Local"
CSV_CATEGORY_COLUMN = "Categoria"

# Valores padrão para campos no JSON enriquecido quando não encontrados no CSV
DEFAULT_VALUE_FOR_MISSING_CSV_DATA = None
# --------------------------------------------------------

def normalizar_genero(genero_str):
    """Normaliza a string de gênero para 'male', 'female', ou 'unknown'."""
    if not genero_str or not isinstance(genero_str, str):
        return "unknown"
    genero_lower = genero_str.strip().lower()
    if genero_lower == "masculino" or genero_lower == "male":
        return "male"
    elif genero_lower == "feminino" or genero_lower == "female":
        return "female"
    return "unknown"

def carregar_json_original(filepath):
    """Carrega o arquivo JSON original."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo JSON de entrada '{filepath}' não encontrado.")
        return None
    except json.JSONDecodeError as e:
        print(f"Erro: Arquivo JSON '{filepath}' não é um JSON válido. Detalhe: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado ao carregar o arquivo JSON '{filepath}': {e}")
        return None

def transformar_json_para_lista_de_objetos(json_data_original):
    """
    Transforma o JSON original (formato colunar) em uma lista de objetos.
    Usa a chave 'bib' como referência principal para os índices.
    Preserva o gênero original do campo `demographic.gender.label` para comparação posterior.
    """
    if not json_data_original:
        print("Dados JSON originais estão vazios ou não foram carregados.")
        return []
    lista_objetos = []
    if "bib" not in json_data_original or not isinstance(json_data_original["bib"], dict):
        print("Erro: Formato do JSON de entrada inválido. Chave 'bib' não encontrada ou não é um dicionário.")
        return []

    indices_str = [idx for idx in json_data_original["bib"].keys() if idx.isdigit()]
    if not indices_str:
        print("Erro: Não foram encontrados índices numéricos válidos na chave 'bib' do JSON.")
        return []
    
    indices_ordenados_str = sorted(indices_str, key=int)
        
    for idx_str in indices_ordenados_str:
        bib_entry = json_data_original["bib"].get(idx_str)
        if not isinstance(bib_entry, dict) or bib_entry.get("number") is None:
            continue

        objeto_linha = {}
        genero_demographic_original = "unknown"
        demographic_data = json_data_original.get("demographic", {}).get(idx_str)
        if isinstance(demographic_data, dict) and isinstance(demographic_data.get("gender"), dict):
            genero_demographic_original = normalizar_genero(demographic_data["gender"].get("label"))
        objeto_linha["_genero_json_original_demographic"] = genero_demographic_original

        for chave_principal, dados_coluna in json_data_original.items():
            if isinstance(dados_coluna, dict):
                objeto_linha[chave_principal] = dados_coluna.get(idx_str)
        
        lista_objetos.append(objeto_linha)

    print(f"JSON original transformado em {len(lista_objetos)} objetos (baseado na chave 'bib').")
    return lista_objetos

def carregar_dados_csv_como_lookup(filepath, bib_column, rank_column, gender_column, distance_column, local_column, category_column_csv): # Adicionado category_column_csv
    """Carrega dados do CSV e os estrutura em um dicionário para busca rápida pelo bib."""
    lookup_data = {}
    try:
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Erro: O arquivo CSV '{filepath}' parece estar vazio ou não tem cabeçalho.")
                return None
            
            required_cols = {bib_column, rank_column, gender_column, distance_column, local_column, category_column_csv} # Adicionado category_column_csv
            missing_cols = required_cols - set(reader.fieldnames)
            if missing_cols:
                print(f"Erro: Colunas esperadas ausentes no cabeçalho do CSV '{filepath}': {', '.join(missing_cols)}")
                return None

            for row in reader:
                bib = row.get(bib_column)
                if bib:
                    bib_limpo = str(bib).strip()
                    if not bib_limpo:
                        continue

                    distancia_str_csv = row.get(distance_column, "").strip()
                    match = re.search(r'\d+', distancia_str_csv)
                    distancia_num_csv = match.group(0) if match else None
                    
                    genero_csv_original_str = row.get(gender_column, "").strip()
                    genero_csv_normalizado = normalizar_genero(genero_csv_original_str)

                    lookup_data[bib_limpo] = {
                        'rank_csv': row.get(rank_column, DEFAULT_VALUE_FOR_MISSING_CSV_DATA),
                        'gender_csv_normalizado': genero_csv_normalizado,
                        'distance_num_csv': distancia_num_csv,
                        'distance_str_csv': distancia_str_csv,
                        'local_csv': row.get(local_column, DEFAULT_VALUE_FOR_MISSING_CSV_DATA),
                        'category_csv': row.get(category_column_csv, DEFAULT_VALUE_FOR_MISSING_CSV_DATA) # Lendo a categoria do CSV
                    }
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV de entrada '{filepath}' não encontrado.")
        return None
    except Exception as e:
        print(f"Erro inesperado ao ler o arquivo CSV '{filepath}': {e}")
        return None
    print(f"Carregados {len(lookup_data)} registros do CSV para lookup.")
    return lookup_data

def enriquecer_json_com_csv(lista_json_objetos_com_genero_original, csv_lookup):
    """ETAPA 1: Enriquecer o JSON `modelo` com dados do CSV."""
    if not lista_json_objetos_com_genero_original:
        return []
    
    dados_enriquecidos = []
    bibs_encontrados_csv = 0
    bibs_nao_encontrados_csv = 0

    for item_json_original in lista_json_objetos_com_genero_original:
        novo_item = item_json_original.copy() 
        bib_para_busca = str(item_json_original["bib"].get("number")).strip()
        
        dados_csv = csv_lookup.get(bib_para_busca) if csv_lookup else None

        if dados_csv:
            novo_item['rank'] = dados_csv.get('rank_csv', DEFAULT_VALUE_FOR_MISSING_CSV_DATA)
            novo_item['gender'] = dados_csv.get('gender_csv_normalizado', DEFAULT_VALUE_FOR_MISSING_CSV_DATA)
            novo_item['distance'] = dados_csv.get('distance_num_csv', DEFAULT_VALUE_FOR_MISSING_CSV_DATA)
            novo_item['category'] = dados_csv.get('category_csv', DEFAULT_VALUE_FOR_MISSING_CSV_DATA) # Usando category_csv
            bibs_encontrados_csv += 1
        else: 
            novo_item['rank'] = DEFAULT_VALUE_FOR_MISSING_CSV_DATA
            novo_item['gender'] = DEFAULT_VALUE_FOR_MISSING_CSV_DATA
            novo_item['distance'] = DEFAULT_VALUE_FOR_MISSING_CSV_DATA
            novo_item['category'] = DEFAULT_VALUE_FOR_MISSING_CSV_DATA # Padrão se não encontrado
            bibs_nao_encontrados_csv +=1
        
        dados_enriquecidos.append(novo_item)
    
    print(f"\n--- Resumo da Etapa 1 (Enriquecimento JSON) ---")
    print(f"Total de itens JSON (com N de Peito válido) processados: {len(lista_json_objetos_com_genero_original)}")
    if csv_lookup is not None:
        print(f"N de Peito do JSON encontrados e enriquecidos com dados do CSV: {bibs_encontrados_csv}")
        print(f"N de Peito do JSON não encontrados no CSV: {bibs_nao_encontrados_csv}")
    else:
        print("Lookup do CSV não foi carregado. JSON não pôde ser enriquecido com dados do CSV.")
    return dados_enriquecidos

def gerar_csv_comparacao_e_relatorio_genero(lista_json_com_dados_originais, csv_lookup_original, output_csv_filepath):
    """ETAPA 2: Gera CSV de comparação e relatório de gênero."""
    if not lista_json_com_dados_originais:
        print("Lista de JSON original está vazia. Não é possível gerar CSV de comparação.")
        return
    if not csv_lookup_original:
        print("Dados originais do CSV (csv_lookup_original) não disponíveis. Não é possível gerar CSV de comparação completo.")
        return

    print(f"\n--- Iniciando Etapa 2 (Comparação de Gênero e Geração de CSV) ---")
    
    total_bibs_com_ambos_generos_validos = 0
    generos_coincidentes = 0
    generos_divergentes = 0
    
    try:
        with open(output_csv_filepath, 'w', newline='', encoding='utf-8') as f_out:
            fieldnames = [
                "N de Peito", # Alterado para manter seu padrão
                "Genero JSON",
                "Genero CSV",
                "Distancia CSV",
                "Local CSV",
                "Generos Coincidem"
            ]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for item_json in lista_json_com_dados_originais:
                bib_para_busca = str(item_json["bib"].get("number")).strip()
                dados_csv_atleta = csv_lookup_original.get(bib_para_busca)
                
                if dados_csv_atleta:
                    genero_original_json_demographic = item_json.get("_genero_json_original_demographic", "unknown")
                    genero_csv_normalizado = dados_csv_atleta.get('gender_csv_normalizado', "unknown")
                    
                    status_coincidencia = "Não Comparável"
                    if genero_original_json_demographic != "unknown" and genero_csv_normalizado != "unknown":
                        total_bibs_com_ambos_generos_validos += 1
                        if genero_original_json_demographic == genero_csv_normalizado:
                            generos_coincidentes += 1
                            status_coincidencia = "Sim"
                        else:
                            generos_divergentes +=1
                            status_coincidencia = "Não"
                    elif genero_original_json_demographic != "unknown" or genero_csv_normalizado != "unknown":
                        status_coincidencia = "Parcialmente Informado"

                    writer.writerow({
                        "N de Peito": bib_para_busca, # Alterado para manter seu padrão
                        "Genero JSON": genero_original_json_demographic,
                        "Genero CSV": genero_csv_normalizado,
                        "Distancia CSV": dados_csv_atleta.get('distance_str_csv', ""),
                        "Local CSV": dados_csv_atleta.get('local_csv', ""),
                        "Generos Coincidem": status_coincidencia
                    })

        print(f"Arquivo de comparação '{output_csv_filepath}' gerado com sucesso.")
    except Exception as e:
        print(f"Erro ao gerar o arquivo CSV de comparação '{output_csv_filepath}': {e}")
        return

    print(f"\n--- Relatório de Comparação de Gênero (JSON vs CSV) ---")
    if total_bibs_com_ambos_generos_validos > 0:
        print(f"Total de N de Peito com gênero válido em ambas as fontes e comparados: {total_bibs_com_ambos_generos_validos}")
        print(f"Gêneros Coincidentes: {generos_coincidentes}")
        print(f"Gêneros Divergentes: {generos_divergentes}")
        percentual_acerto = (generos_coincidentes / total_bibs_com_ambos_generos_validos) * 100
        print(f"Percentual de Coincidência de Gênero (entre os comparáveis): {percentual_acerto:.2f}%")
    else:
        print("Nenhum registro pôde ser comparado para gênero por falta de informações.")


def salvar_json_enriquecido(filepath, data_list):
    """Salva o JSON enriquecido em um arquivo JSON formatado."""
    data_para_salvar = []
    for item in data_list:
        item_sem_temp = item.copy()
        item_sem_temp.pop("_genero_json_original_demographic", None)
        data_para_salvar.append(item_sem_temp)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_para_salvar, f, ensure_ascii=False, indent=4)
        print(f"JSON enriquecido salvo com sucesso em '{filepath}'")
    except Exception as e:
        print(f"Erro ao salvar o arquivo JSON '{filepath}': {e}")

# --- Função Principal ---
def main():
    print("Iniciando processo de combinação e comparação de dados...")

    json_original_data = carregar_json_original(JSON_INPUT_FILE)
    if json_original_data is None:
        print("Falha ao carregar JSON de entrada. Encerrando.")
        return

    lista_json_com_dados_originais = transformar_json_para_lista_de_objetos(json_original_data)
    if not lista_json_com_dados_originais:
        print("Transformação do JSON original falhou ou resultou em lista vazia. Encerrando.")
        return

    dados_csv_lookup = carregar_dados_csv_como_lookup(
        CSV_INPUT_FILE,
        CSV_BIB_COLUMN, CSV_RANK_COLUMN, CSV_GENDER_COLUMN, 
        CSV_DISTANCE_COLUMN, CSV_LOCAL_COLUMN, CSV_CATEGORY_COLUMN # Passando a nova constante
    )
    
    print(f"\n--- Iniciando Etapa 1: Enriquecimento do JSON ---")
    json_enriquecido = enriquecer_json_com_csv(lista_json_com_dados_originais, dados_csv_lookup)

    if json_enriquecido:
        salvar_json_enriquecido(JSON_OUTPUT_FILE, json_enriquecido)
    else:
        print("JSON não foi enriquecido (lista vazia ou falha no processo).")

    if lista_json_com_dados_originais and dados_csv_lookup:
         gerar_csv_comparacao_e_relatorio_genero(lista_json_com_dados_originais, dados_csv_lookup, CSV_COMPARISON_FILE)
    else:
        print("\nNão foi possível prosseguir para a Etapa 2 (comparação de gênero e geração de CSV).")
        if not lista_json_com_dados_originais:
            print("  Motivo: A lista de objetos JSON original (baseada na chave 'bib') está vazia ou não foi carregada.")
        if not dados_csv_lookup:
            print("  Motivo: Os dados do CSV de atletas não foram carregados.")
            
    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()
