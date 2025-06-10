# Bibliotecas Necessárias

* Python - Linguagem principal do projeto
* R - Linguagem secundária do projeto
* rpy2 - Biblioteca que permite a integração entre Python e R
* river - Framework para detecção de drift em tempo real
* pandas - Manipulação e análise de dados
* numpy - Operações numéricas com arrays e matrizes
* multiprocessing - Execução paralela de tarefas.
* itertools - Gerar combinações e permutações de features e detectores

---

# Sobre os Arquivos

Para a aplicação da metodologia, foram desenvolvidos alguns scripts baseados em diferentes conceitos ensinados na disciplina e disponibilizados no Harbinger. Como mencionado em aula, utilizei uma estratégia que envolvesse tanto Python quanto R, pois existem funcionalidades exclusivas em cada linguagem. No entanto, toda a aplicação está desenvolvida principalmente em Python, com algumas chamadas ao R, como para o Fuzzy Ensemble DD (FEDD - Artigo IJCNN Lucas) e o softED (Harbinger).

### Estrutura dos Arquivos:

* **./data**:
  Contém os dados disponibilizados no repositório GitHub do StatsBomb. Obs.: não utilizei todos os dados, apenas os arquivos referentes à competição e temporadas específicas da La Liga 2015/2016. Esse arquivo foi omitido, pois possui um tamanho de 9GB, sendo o mesmo obtido no [Statsbomb](https://github.com/statsbomb/open-data).

* **./doc**: 
  Contém toda a documentação dos dados disponibilizados no repositório GitHub do StatsBomb, que foi útil para orientar a criação do dataset principal utilizado na metodologia.

* **./scripts/df\_creation.ipynb**:
  Notebook com o pipeline de desenvolvimento do dataset principal, contendo os eventos e as features de todas as partidas para todos os times da La Liga 2015/2016. Inclui também o evento do gol com aplicação de *lag*, necessário para a avaliação com o softED na metodologia.

* **./scripts/df\_matches.csv**:
  Dataset principal criado a partir do notebook `df_creation.ipynb`, que será utilizado na metodologia.

* **./scripts/dd\_original.py**:
  Script da metodologia principal de detecção online, utilizando as séries temporais das features do time defensivo como forma de previsão de gols sofridos. Utiliza combinações de detectores e features separadas. As funções empregam conceitos como: detecção de *drift* com Page-Hinley (com otimização de seus parâmetros) e avaliação com o softED do Harbinger.

* **./scripts/dd\_ensemble.py**:
  Script da metodologia principal de detecção online utilizando *ensemble* de features, além de combinações de detectores. Aplica conceitos como: detecção de *drift* com Page-Hinley (com otimização de parâmetros), *ensemble* de features com o FEDD (Fuzzy Ensemble Drift Detection) e avaliação com o softED do Harbinger.

* **./results**:
  Contém todos os resultados e os melhores desempenhos obtidos com a metodologia, tanto utilizando features separadas quanto com *ensemble* de features. Eu mantive apenas as melhores combinações, por conta dos outros excederem o espaço de carregamento do github.

---

# Instruções de Uso

0. Instale as bibliotecas necessárias
1. Execute o notebook `df_creation.ipynb` para gerar o dataset principal a ser utilizado na metodologia.
2. Execute a metodologia com features separadas utilizando `dd_original.py`, e depois com *ensemble* de features utilizando `dd_ensemble.py`.
3. Analise os resultados em:

   * `/results/df_best_results_original.xlsx` (features separadas)
   * `/results/df_best_results_ensemble.xlsx` (*ensemble* de features)

