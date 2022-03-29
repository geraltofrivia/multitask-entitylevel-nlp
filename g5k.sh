# #############
# File used to run experiments on G5k
# #############

oarsub -l walltime=8 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d ontonotes -t ner -enc bert-base-uncased -e 5 -dv cuda"