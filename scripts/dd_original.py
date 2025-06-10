import pandas as pd
from rpy2.robjects import r, pandas2ri
from river import drift
from itertools import product
from multiprocessing import Pool
import time
import json

# Função para detecção de drift com PageHinkley (Usei no python pois o Heimdall não considera drift pra cima e pra baixo)
def drift_detection(stream_column, drifter_name, params):
    
    drifter = drift.PageHinkley(**params)

    drifts_idx = []
    for val in stream_column:
        drifter.update(val)
        drifts_idx.append(drifter.drift_detected)
    return drifts_idx

# Função que retorna os scores com avaliação softED (Código adaptado do harbinger)
def har_eval_soft(df_match, analysed_team, k):
    # Ativa conversão automática pandas <-> R
    pandas2ri.activate()
    
    # Converte para R dataframe
    r_df_match = pandas2ri.py2rpy(df_match)

    # Chama avaliação no R
    r.assign("df_match", r_df_match)
    r(f'''
    library(harbinger)
    library(daltoolbox)
    
    # usa os dados do time da casa para avaliar, se o time analisado for casa
    if (any(df_match$home_team == "{analysed_team}")) {{
    
        evaluation <- evaluate(har_eval_soft(sw_size = {k}), df_match$drift_casa, df_match$lag_gol_fora)

    # usa os dados do time de fora para avaliar, se o time analisado for fora
    }} else if (any(df_match$away_team == "{analysed_team}")) {{
    
        evaluation <- evaluate(har_eval_soft(sw_size = {k}), df_match$drift_fora, df_match$lag_gol_casa)            
    
    }}
    ''')

    # Extrai métricas
    TP, FP, FN, TN = r('evaluation$TP')[0], r('evaluation$FP')[0], r('evaluation$FN')[0], r('evaluation$TN')[0]

    return TP, FP, FN, TN

# Função principal para calcular os resultados de todas as combinações de hiperparâmetros e features para um time específico
def team_matches_drift_detection(args):

    df_matches, analysed_team, products, k = args
    
    analysed_team_results = []

    analysed_team_matches = df_matches.loc[(df_matches['home_team'] == analysed_team) | (df_matches['away_team'] == analysed_team)].copy()

    # Loop para cada partida do time específico analisado
    for match_id in analysed_team_matches['match_id'].unique():

        df_match = df_matches[df_matches['match_id'] == match_id].copy()    

        # Loop para possibilidade de parâmetros no products
        for product_ in products:

            feature_casa = product_['feature']['feature_values'][0]
            feature_fora = product_['feature']['feature_values'][1]

            detector_name = product_['detector_combinations']['detector_name']
            detector_params = product_['detector_combinations']['params']

            df_match['drift_casa'] = drift_detection(df_match[feature_casa], detector_name, detector_params)
            df_match['drift_fora'] = drift_detection(df_match[feature_fora], detector_name, detector_params)
            
            # Tipagens
            df_match['gol_fora'] = df_match['gol_fora'].astype(bool)
            df_match['lag_gol_fora'] = df_match['lag_gol_fora'].astype(bool)

            df_match['gol_casa'] = df_match['gol_casa'].astype(bool)
            df_match['lag_gol_casa'] = df_match['lag_gol_casa'].astype(bool)

            # Extrai métricas
            TP, FP, FN, TN = har_eval_soft(df_match, analysed_team, k)

            # Armazena resultados
            analysed_team_results.append({
                "match_id": df_match['match_id'].iloc[0],
                "events_df": df_match,
                "analysed_team": analysed_team, 
                "home_team": df_match['home_team'].iloc[0],
                "away_team": df_match['away_team'].iloc[0],
                "home_score": df_match['home_score'].iloc[0],
                "away_score": df_match['away_score'].iloc[0],
                "feature": product_['feature']['feature_name'],
                "detector": detector_name,
                "drifter_params": detector_params,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN              
            })

    # Transformando todos os resultados do time analisado em um dataframe
    df_results = pd.DataFrame(analysed_team_results)

    print(f'Processamento concluído para o time {analysed_team}')

    return df_results

# Função que retorna as combinações de hiperparâmetros do detector e as features criadas
def get_products():

    features = [{'feature_name': 'Passes Errados', 'feature_values': ['pass_unsuccessful_casa', 'pass_unsuccessful_fora']},
                {'feature_name': 'Erros', 'feature_values': ['errors_casa', 'errors_fora']},
                {'feature_name': 'DA', 'feature_values': ['DA_casa', 'DA_fora']},
                {'feature_name': 'PPDA', 'feature_values': ['PPDA_casa', 'PPDA_fora']},
                {'feature_name': 'PPSR', 'feature_values': ['PPSR_casa', 'PPSR_fora']}        
                ]
    
    # Parâmetros para o Page Hinkley
    page_hinkley_params = {
        'min_instances': [2, 5, 10],
        'threshold': [1, 3, 5, 7, 10],
        'delta': [0.001, 0.005, 0.01],
        'alpha': [0.9, 0.95, 0.99],
        'mode': ['up', 'down']
    }

    # Gerar combinações para o Page Hinkley
    page_hinkley_combinations = product(
        page_hinkley_params['min_instances'],
        page_hinkley_params['threshold'],
        page_hinkley_params['delta'],
        page_hinkley_params['alpha'],
        page_hinkley_params['mode']
    )

    # Lista para armazenar as combinações de detectores com parâmetros
    detector_combinations = []

    # Adicionar combinações para o Page Hinkley
    for combo in page_hinkley_combinations:
        param_dict = {
            'detector_name': 'PageHinkley',
            'params': {
                'min_instances': combo[0],
                'threshold': combo[1],
                'delta': combo[2],
                'alpha': combo[3],
                'mode': combo[4]
            }
        }
        detector_combinations.append(param_dict)

    products = [dict(zip(('detector_combinations', 'feature'), values))
                for values in product(detector_combinations, features)]   
    
    return products

# Aplicação principal do código que gera todos os resultados (Fiz paralelização em 5 para rodar mais rápido)
if __name__ == '__main__':

    # Parâmetros
    k = 10 # timepoint pra acrescentar ou diminuir no ponto hard do softED (mesmo tamanho do lag_size)

    teams = ['Granada', 'Getafe', 'Real Madrid', 'Valencia', 'Sporting Gijón',
           'Espanyol', 'Rayo Vallecano', 'Málaga', 'Levante UD', 'Las Palmas',
           'Barcelona', 'Eibar', 'Villarreal', 'Real Sociedad',
           'RC Deportivo La Coruña', 'Real Betis', 'Athletic Club',
           'Atlético Madrid', 'Celta Vigo', 'Sevilla']

    products = get_products()

    # Carrega os dados
    df_matches = pd.read_csv('./scripts/df_matches.csv')

    # Realiza o processamento
    inicio = time.time()
    with Pool(processes=5) as pool:
            results = pool.map(team_matches_drift_detection, [(df_matches, analysed_team, products, k) for analysed_team in teams])
    fim = time.time()
    print(f'Tempo total de execução: {round(fim - inicio)} segundos.')

    df_results_total = pd.concat(results)
    df_results_total.reset_index(drop=True, inplace=True)
    df_results_total.to_pickle('./results/df_results.pkl')

    # Transformando o dicionário de params em json para poder servir como chave no groupby
    df_results_total['drifter_params'] = df_results_total['drifter_params'].apply(json.dumps)

    # Agregrando pela soma de TPs, FPs e FNs de todas as combinações nas partidas do time analisado (casa e fora) + features + detector e seu conjunto de parametros
    df_results_agg = (
    df_results_total
    .groupby(['analysed_team', 'feature', 'detector', 'drifter_params']) 
    .agg(
        TP_sum=('TP', 'sum'),
        FP_sum=('FP', 'sum'),
        FN_sum=('FN', 'sum')
    ).round(4)
    )

    # Cálculo das métricas principais a partir dos TPs, FPs e FNs totais
    df_results_agg['precision_total'] = round(df_results_agg['TP_sum'] / (df_results_agg['TP_sum'] + df_results_agg['FP_sum']), 4)
    df_results_agg['recall_total'] = round(df_results_agg['TP_sum'] / (df_results_agg['TP_sum'] + df_results_agg['FN_sum']), 4)
    df_results_agg['f1_total'] = round(2 * df_results_agg['precision_total'] * df_results_agg['recall_total'] / (df_results_agg['precision_total'] + df_results_agg['recall_total']), 4)

    # Tratar divisões por zero (NaNs) no final
    df_results_agg = df_results_agg.fillna(0).reset_index().sort_values('f1_total', ascending=False)

    # Seleciona o índice da melhor linha (maior F1) por time
    best_rows = df_results_agg.groupby(['analysed_team'])['f1_total'].idxmax()

    # Usa os índices para criar o novo DataFrame com as melhores linhas
    df_best = df_results_agg.loc[best_rows].reset_index(drop=True)
    df_best.sort_values('f1_total', ascending=False).to_excel('./results/df_best_results.xlsx')