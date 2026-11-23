IRACE — CALIBRAÇÃO DA ILS SEM PATH RELINKING (FI/BI)
=====================================================

Localização:
  parcel-locker-mnl-ils-qtla/calibration/irace_ils_without_PR/

Arquivos necessários fora desta pasta:
  ../../src/mnl_ils.py
  ../../instances/instances_seed_11_12_13_N60_100_200_400/

A lista instances_calibration.txt tem 72 arquivos .npz, em 18 blocos
consecutivos de quatro tamanhos N=60,100,200,400.
Instâncias: seeds 11,12,13; lambda 0.5,1.0,2.0;
p=3/6,5/10,10/20,20/30 por tamanho, respectivamente.

A/S1/S2 habilitados, Path Relinking desabilitado. O irace calibra
13 parâmetros, incluindo firstImprovement (0=BI, 1=FI).
O cenário preserva maxExperiments=5000, seed=12345, teste de Friedman,
blockSize=4, parallel=1 e demais parâmetros originais.

Para executar em cópia limpa do experimento:
  1. Configure o RStudio com o diretório de trabalho
     calibration/irace_ils_without_PR.
  2. Confirme que `python` no PATH é o ambiente correto, com NumPy,
     dependências do solver e os módulos do projeto.
  3. Execute source("01_preflight.R").
  4. Somente após sucesso, execute source("02_run_irace.R").

IMPORTANTE: não execute na pasta que já contém os resultados originais.
O script de calibração impede a substituição dos resultados históricos.
Os nomes irace_ils_calibration_fi_bi_4groups.Rdata e demais saídas
foram preservados de propósito para rastreabilidade científica.
A ordem da lista e os 13 parâmetros não foram modificados.
Os scripts ajustados devem ser colocados nesta pasta com seus nomes
originais: 01_preflight.R, 02_run_irace.R, scenario.txt e target_runner_ils.py.
Não é necessário executar o antigo build_instances_calibration.py:
as instâncias já estão disponíveis em instances/.
