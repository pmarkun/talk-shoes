import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
import concurrent.futures
from tqdm import tqdm # Importa a biblioteca para a barra de progresso

# --- Configurações ---
# Número de threads a serem usadas para as requisições paralelas.
# Aumentar este número pode acelerar o processo, mas não exagere (entre 10-20 é um bom começo).
MAX_WORKERS = 15 

# Cabeçalho para simular um navegador e evitar bloqueios
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_and_parse_page(url: str, headers: dict) -> list:
    """
    Função 'worker' que busca o HTML de uma única página e extrai as linhas da tabela.
    Projetada para ser executada em um thread separado.

    Args:
        url (str): A URL da página a ser raspada.
        headers (dict): Os cabeçalhos da tabela, para garantir o número correto de colunas.

    Returns:
        list: Uma lista de listas, onde cada lista interna representa uma linha de dados.
              Retorna uma lista vazia em caso de erro.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = 'ISO-8859-1'
        page_soup = BeautifulSoup(response.text, 'html.parser')
        
        data_rows = page_soup.find_all('tr', {'bgcolor': '#FFFFFF'})
        page_results = []
        for row in data_rows:
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cols) == len(headers):
                page_results.append(cols)
        return page_results
    except requests.exceptions.RequestException as e:
        # Silenciosamente ignora páginas que falham para não parar todo o processo
        # print(f"\nErro ao buscar {url}: {e}") # Descomente para depurar
        return []

def fetch_prova(start_url: str):
    """
    Raspa os resultados de todas as páginas de uma prova de forma paralela.

    Args:
        start_url (str): A URL da primeira página de resultados.

    Returns:
        pd.DataFrame: Um DataFrame do pandas contendo todos os resultados da prova.
    """
    print(f"Iniciando a raspagem para a URL: {start_url}")
    
    try:
        # --- Passo 1: Acessar a primeira página para obter metadados (não paralelo) ---
        response = requests.get(start_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = 'ISO-8859-1'
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Passo 2: Extrair o número total de páginas e cabeçalhos ---
        page_info_container = soup.find(lambda tag: 'EXIBINDO PÁGINA' in tag.get_text())
        if not page_info_container:
            print("Não foi possível encontrar o container da informação de paginação. Abortando.")
            return pd.DataFrame()

        total_pages_match = re.search(r'DE\s*(\d+)', page_info_container.get_text(strip=True))
        if not total_pages_match:
            print("Não foi possível extrair o número total de páginas. Abortando.")
            return pd.DataFrame()
            
        total_pages = int(total_pages_match.group(1))
        print(f"Encontrado um total de {total_pages} páginas.")

        header_row = soup.find('tr', {'bgcolor': '#EFEFEF'})
        if not header_row:
            print("Não foi possível encontrar o cabeçalho da tabela. Abortando.")
            return pd.DataFrame()
        
        headers = [th.get_text(strip=True) for th in header_row.find_all('td')]
        print(f"Cabeçalhos encontrados: {headers}")

        # --- Passo 3: Processar a primeira página e preparar para as demais ---
        all_results = []
        # Processa as linhas da primeira página já carregada
        first_page_rows = soup.find_all('tr', {'bgcolor': '#FFFFFF'})
        for row in first_page_rows:
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cols) == len(headers):
                all_results.append(cols)

        # Criar a lista de URLs para as páginas restantes
        urls_to_fetch = [
            re.sub(r'PaginaAtual=\d+', f'PaginaAtual={page_num}', start_url)
            for page_num in range(2, total_pages + 1)
        ]

        # --- Passo 4: Executar a raspagem paralela com barra de progresso ---
        print(f"Raspando {len(urls_to_fetch)} páginas restantes com {MAX_WORKERS} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Cria uma função parcial para passar o argumento 'headers' fixo para o worker
            from functools import partial
            worker_fn = partial(fetch_and_parse_page, headers=headers)
            
            # `executor.map` aplica a função a cada URL e retorna os resultados
            # `tqdm` envolve o iterador para mostrar o progresso
            results_iterator = executor.map(worker_fn, urls_to_fetch)
            
            # Envolve o `results_iterator` com `tqdm`
            for page_result in tqdm(results_iterator, total=len(urls_to_fetch), desc="Progresso da Raspagem"):
                if page_result:
                    all_results.extend(page_result)
        
        print(f"Raspagem concluída. Total de {len(all_results)} resultados coletados.")
        # Cria o DataFrame a partir da lista de listas e dos cabeçalhos
        return pd.DataFrame(all_results, columns=headers)

    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão ao acessar a URL inicial {start_url}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return pd.DataFrame()


def main():
    """
    Função principal para orquestrar a raspagem e salvar o arquivo Excel.
    """
    provas_a_raspar = [
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?evento_yescom_id=2456&tipo=3&tipo_do_evento_id=9023&PaginaAtual=1&sexo=M",
            "sheet": "42km_masculino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?evento_yescom_id=2456&tipo=4&tipo_do_evento_id=9023&PaginaAtual=1&sexo=F",
            "sheet": "42km_feminino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9025&tipo=3&evento_yescom_id=2456",
            "sheet": "10km_masculino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9025&tipo=4&evento_yescom_id=2456",
            "sheet": "10km_feminino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9024&tipo=3&evento_yescom_id=2456",
            "sheet": "21km_masculino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9024&tipo=4&evento_yescom_id=2456",
            "sheet": "21km_feminino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9026&tipo=3&evento_yescom_id=2456",
            "sheet": "5km_masculino"
        },
        {
            "url": "https://www.yescom.com.br/codigo_comum/classificacao/codigo/p_classificacao03_v1.asp?tipo_do_evento_id=9026&tipo=4&evento_yescom_id=2456",
            "sheet": "5km_feminino"
        }

    ]
    
    output_filename = "resultados_maratona_sp_2024.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for prova in provas_a_raspar:
            print("-" * 50)
            df_resultados = fetch_prova(prova['url'])
            
            if not df_resultados.empty:
                df_resultados.to_excel(writer, sheet_name=prova['sheet'], index=False)
                print(f"Dados da planilha '{prova['sheet']}' foram adicionados ao arquivo.")
            else:
                print(f"Não foi possível obter dados para a planilha '{prova['sheet']}'.")
    
    print("-" * 50)
    print(f"Processo finalizado. O arquivo '{output_filename}' foi salvo com sucesso!")

if __name__ == "__main__":
    main()