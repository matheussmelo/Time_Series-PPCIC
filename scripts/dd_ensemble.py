import pandas as pd
from rpy2.robjects import r, pandas2ri
from river import drift
from itertools import product
from multiprocessing import Pool
import rpy2.robjects as robjects
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

# Função que retorna uma lista de valores TRUE/FALSE de acordo com o que foi feito no artigo IJCNN do Lucas
def fedd(df: pd.DataFrame, time: str, tolerance: int) -> pd.Series:
    
    pandas2ri.activate()

    # Envia o dataframe pandas para R
    robjects.globalenv['df'] = pandas2ri.py2rpy(df)
    robjects.globalenv['tolerance'] = tolerance
    robjects.globalenv['time'] = time

    # Código R inteiro como string
    r_code = """
    library(daltoolbox)
    library(harbinger)

    har_fuzzify_detections_triangle <- function(value, tolerance) {
      type <- attr(value, "type")
      value <- as.double(value)
      
      if (!tolerance) {
        attr(value, "type") <- type
        return(value)
      }
      
      idx <- which(value >= 1)
      
      if (length(idx) == 0) {
        return(rep(0, length(value)))
      }
      
      lista_series <- lapply(idx, function(i) {
        temp <- rep(FALSE, length(value))
        temp[i] <- TRUE
        temp
      })
      
      detections <- as.data.frame(do.call(cbind, lista_series))
      
      fuzzy_detections <- c()
      
      for (detection in detections){
        detection <- as.double(detection)
        i <- which(detection >= 1)
        n <- length(detection)
        ratio <- 1/tolerance
        range <- tolerance-1
        
        curtype <- ""
        if (!is.null(type))
          curtype <- type[i]
        
        for (j in 1:range) {
          if (i + j <= n) {
            detection[i + j] <- detection[i + j] + (tolerance - j) * ratio
            type[i + j] <- curtype
          }
          if (i - j >= 1) {
            detection[i - j] <- detection[i - j] + (tolerance - j) * ratio
            type[i - j] <- curtype
          }
        }
        
        fuzzy_detections <- cbind(fuzzy_detections, detection)
        attr(value, "type") <- type
      }
      
      fuzzy_detections <- as.data.frame(fuzzy_detections)
      fuzzy_detections$max_value <- apply(fuzzy_detections, 1, max)
      
      return(fuzzy_detections$max_value)
    }

    calculate_fuzzy <- function(df, features, tolerance){
      fuzzy_df <- c()
      drifters_df <- c()
      
      for (feature in features){
        drifters_df <- cbind(drifters_df, df[[feature]])
        ens <- as.data.frame(har_fuzzify_detections_triangle(df[[feature]], tolerance=tolerance))
        fuzzy_df <- cbind(fuzzy_df, ens[,1])
      }
      
      return(list(drifters_df = drifters_df, fuzzy_df = fuzzy_df))
    }

    features <- names(df)[grepl(paste0("^drift.*", time, "$"), names(df))]
    df[features] <- lapply(df[features], function(x) x %in% c("True", TRUE))
    
    output <- calculate_fuzzy(df, features, tolerance=tolerance)

    drifters_df <- output[[1]]
    colnames(drifters_df) <- colnames(df[features])
    
    fuzzy_df <- output[[2]]
    colnames(fuzzy_df) <- colnames(df[features])

    drifters_check <- as.data.frame(drifters_df)
    fuzzy_check <- as.data.frame(fuzzy_df)

    fuzzy_tv_output <- as.data.frame(rep(FALSE, nrow(drifters_df)))
    rownames(fuzzy_tv_output) <- rownames(drifters_check)
    names(fuzzy_tv_output) <- c('drift')

    for (i in 1:nrow(fuzzy_df)){
      row <- fuzzy_check[rownames(fuzzy_check) == i, ]
      if (sum(row) >= 3){
        fuzzy_tv_output[rownames(fuzzy_tv_output) == i, 'drift'] <- TRUE
        drifters_check <- drifters_check[rownames(drifters_check) %in% as.character(i:nrow(drifters_df)),]
        fuzzy_check <- fuzzy_check[rownames(fuzzy_check) %in% as.character(i:nrow(fuzzy_df)),]
        drift_columns <- names(fuzzy_check[,row != 0])
        
        fuzzy_check <- c()
        
        for (feat in names(drifters_check)){
          if (feat %in% drift_columns){
            drift_index <- as.integer(rownames(head(drifters_check[drifters_check[, feat], feat, drop=FALSE], 1)))
            drifters_check[rownames(drifters_check) %in% c(drift_index), feat] <- FALSE
          }
          
          ens <- as.data.frame(har_fuzzify_detections_triangle(drifters_check[[feat]], tolerance=tolerance))
          fuzzy_values <- ens[,1]
          attr(fuzzy_values, 'type') <- NULL
          fuzzy_check <- cbind(fuzzy_check, fuzzy_values)
        }
        
        fuzzy_check <- as.data.frame(fuzzy_check)
        rownames(fuzzy_check) <- rownames(drifters_check)
        names(fuzzy_check) <- names(drifters_check)
      }
    }
    """

    # Executa o código R
    robjects.r(r_code)

    # Pega o vetor drift diretamente
    fuzzy_tv_output = robjects.r['fuzzy_tv_output']

    # Converte para pandas Series
    drift_series = list(fuzzy_tv_output.rx2('drift'))
    
    return drift_series

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
    
        evaluation <- evaluate(har_eval_soft(sw_size = {k}), df_match$fedd_casa, df_match$lag_gol_fora)

    # usa os dados do time de fora para avaliar, se o time analisado for fora
    }} else if (any(df_match$away_team == "{analysed_team}")) {{
    
        evaluation <- evaluate(har_eval_soft(sw_size = {k}), df_match$fedd_fora, df_match$lag_gol_casa)            
    
    }}
    ''')

    # Extrai métricas
    TP, FP, FN, TN = r('evaluation$TP')[0], r('evaluation$FP')[0], r('evaluation$FN')[0], r('evaluation$TN')[0]

    return TP, FP, FN, TN

# Função principal para calcular os resultados de todas as combinações de hiperparâmetros para um time específico
def team_matches_drift_detection(args):

    df_matches, analysed_team, features, products, k, tolerance = args
    
    analysed_team_results = []

    analysed_team_matches = df_matches.loc[(df_matches['home_team'] == analysed_team) | (df_matches['away_team'] == analysed_team)].copy()

    # Loop para cada partida do time específico analisado
    for match_id in analysed_team_matches['match_id'].unique():

        df_match = df_matches[df_matches['match_id'] == match_id].copy()    

        # Loop para possibilidade de parâmetros no products
        for product_ in products:

            detector_name = product_['detector_name']
            detector_params = product_['params']

            for feature in features:
                feature_casa = feature['feature_values'][0]
                feature_fora = feature['feature_values'][1]

                # faz a detecção para as features do time da casa
                df_match[f'drift_{feature_casa}'] = drift_detection(df_match[feature_casa], detector_name, detector_params)

                # faz a detecção para as features do time de fora
                df_match[f'drift_{feature_fora}'] = drift_detection(df_match[feature_fora], detector_name, detector_params)
              
            df_match['fedd_casa'] = fedd(df_match, 'casa', tolerance)
            df_match['fedd_fora'] = fedd(df_match, 'fora', tolerance)

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
                "analysed_team": analysed_team, 
                "home_team": df_match['home_team'].iloc[0],
                "away_team": df_match['away_team'].iloc[0],
                "home_score": df_match['home_score'].iloc[0],
                "away_score": df_match['away_score'].iloc[0],
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

# Função para obter as combinações de hiperparâmetros do detector
def get_products():
    
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

    return detector_combinations

# Aplicação principal do código que gera todos os resultados (Fiz paralelização em 5 para rodar mais rápido)
if __name__ == '__main__':

    features = [{'feature_values': ['pass_unsuccessful_casa', 'pass_unsuccessful_fora']},
                {'feature_values': ['errors_casa', 'errors_fora']},
                {'feature_values': ['DA_casa', 'DA_fora']},
                {'feature_values': ['PPDA_casa', 'PPDA_fora']},
                {'feature_values': ['PPSR_casa', 'PPSR_fora']}        
                ]

    # Parâmetros
    k = 10 # timepoint pra acrescentar ou diminuir no ponto hard do softED (mesmo tamanho do lag_size)

    teams = ['Granada', 'Getafe', 'Real Madrid', 'Valencia', 'Sporting Gijón',
           'Espanyol', 'Rayo Vallecano', 'Málaga', 'Levante UD', 'Las Palmas',
           'Barcelona', 'Eibar', 'Villarreal', 'Real Sociedad',
           'RC Deportivo La Coruña', 'Real Betis', 'Athletic Club',
           'Atlético Madrid', 'Celta Vigo', 'Sevilla']

    products = get_products()

    tolerance = 10

    # Carrega os dados
    df_matches = pd.read_csv('./scripts/df_matches.csv')

    inicio = time.time()    

    with Pool(processes=5) as pool:
            results = pool.map(team_matches_drift_detection, [(df_matches, analysed_team, features, products, k, tolerance) for analysed_team in teams])

    fim = time.time()

    print(f'Tempo total de execução: {round(fim - inicio)} segundos.')

    df_results_total = pd.concat(results)
    df_results_total.reset_index(drop=True, inplace=True)
    df_results_total.to_excel('./results/df_results_ensemble.xlsx')

    # Transformando o dicionário de params em json para poder servir como chave no groupby
    df_results_total['drifter_params'] = df_results_total['drifter_params'].apply(json.dumps)

    # Agregrando pela soma de TPs, FPs e FNs de todas as combinações nas partidas do time analisado (casa e fora) + features + detector e seu conjunto de parametros
    df_results_agg = (
    df_results_total
    .groupby(['analysed_team', 'detector', 'drifter_params']) 
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
    df_best.sort_values('f1_total', ascending=False).to_excel('./results/df_best_results_ensemble.xlsx')

    