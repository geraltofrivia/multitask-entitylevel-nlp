# #############
# File used to run experiments on G5k
# #############

oarsub -l gpu=1,walltime=10,core=4 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d ontonotes -t ner -enc bert-base-uncased -e 200 -dv cuda -wb"