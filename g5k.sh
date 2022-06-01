# #############
# File used to run experiments on G5k
# #############

#oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d ontonotes -t ner -t pruner -t coref -enc bert-base-uncased -e 200 -dv cuda -wb -wbm mtl_should_run_this_time"
#oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm first"
#python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm "first scierc run with default things"

oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --trim --debug --filter-candidates-pos -wb --wandb-name personal-baseline-redo"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-01 -lr 0.01"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-001 -lr 0.001"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-005 -lr 0.005"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-0001 -lr 0.0001"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-0005 -lr 0.0005"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-00001 -lr 0.00001"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-00005 -lr 0.00005"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-000001 -lr 0.000001"
oarsub -l gpu=1,walltime=3 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-000005 -lr 0.000005"