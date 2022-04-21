# #############
# File used to run experiments on G5k
# #############

oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d ontonotes -t ner -t pruner -t coref -enc bert-base-uncased -e 200 -dv cuda -wb -wbm mtl_should_run_this_time"
oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm first"
python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm "first scierc run with default things"