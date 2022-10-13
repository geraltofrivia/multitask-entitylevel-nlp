python src/run.py -d ontonotes -t coref 1.0 False -t pruner 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-crpr-on-trim-50 --train-encoder True -tok bert-base-uncased --trim 50 train
python src/run.py -d ontonotes -t coref 1.0 False -t pruner 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-crpr-on-trim-100 --train-encoder True -tok bert-base-uncased --trim 100 train
python src/run.py -d ontonotes -t coref 1.0 False -t pruner 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-crpr-on-trim-200 --train-encoder True -tok bert-base-uncased --trim 200 train
python src/run.py -d ontonotes -t coref 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-cr-on-trim-50 --train-encoder True -tok bert-base-uncased --trim 50 train
python src/run.py -d ontonotes -t coref 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-cr-on-trim-100 --train-encoder True -tok bert-base-uncased --trim 100 train
python src/run.py -d ontonotes -t coref 1.0 False -enc bert-base-uncased -e 35 -dv cuda -wb --wandb-name sweepds-cr-on-trim-200 --train-encoder True -tok bert-base-uncased --trim 200 train
