python src/run.py -d ontonotes -t coref 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.0 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.1 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.1 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.5 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.5 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 1.0 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw1. --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.1 True -t pruner 0.1 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.1-prlw.1 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.5 True -t pruner 0.1 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.5-prlw.1 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 1.0 True -t pruner 0.1 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw1.-prlw.1 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.1 True -t pruner 0.5 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.1-prlw.5 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 0.5 True -t pruner 0.5 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw.5-prlw.5 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -t ner 1.0 True -t pruner 0.5 True -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepnr-crpr-on-nrlw1.-prlw.5 --train-encoder True -tok bert-base-uncased --trim 200 train
