# #############
# File used to run experiments on G5k
# #############

#oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d ontonotes -t ner -t pruner -t coref -enc bert-base-uncased -e 200 -dv cuda -wb -wbm mtl_should_run_this_time"
#oarsub -l gpu=1,walltime=7 "source ~/.bashrc; conda activate main;cd ~/work/coref;python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm first"
#python src/run.py -d scierc -t ner -enc bert-base-uncased -e 2000 -dv cuda -wb -wbm "first scierc run with default things"


# #############
# Runs to see which LRs work best when just training the model, with a frozen encoder
# #############
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --trim --debug --filter-candidates-pos -msw 30 -wb --wandb-name cr-on-tr-msw30"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-01 -lr 0.01"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-001 -lr 0.001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-005 -lr 0.005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-0001 -lr 0.0001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-0005 -lr 0.0005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-00001 -lr 0.00001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-00005 -lr 0.00005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-000001 -lr 0.000001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-000005 -lr 0.000005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-0000001 -lr 0.0000001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb -msw 30 --wandb-name cr-on-tr-msw30-0000005 -lr 0.0000005"

#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-01 -lr 0.01"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-001 -lr 0.001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-005 -lr 0.005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-0001 -lr 0.0001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-0005 -lr 0.0005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-00001 -lr 0.00001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-00005 -lr 0.00005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-000001 -lr 0.000001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-000005 -lr 0.000005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-0000001 -lr 0.0000001"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-0000005 -lr 0.0000005"



# #############
# Runs to see if there's any sense in trying to train the encoder
# #############
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --trim --debug --filter-candidates-pos -msw 30 -wb --wandb-name cr-on-tr-msw30"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --trim --debug --filter-candidates-pos -wb --wandb-name personal-baseline-trim-trnenc --train-encoder"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --wandb-name cr-on-tr-trnenc -lr 0.000005 --train-encoder"

# ##############
# TRN ENC Runs which seem to work well
# ##############
oarsub -l gpu=1,walltime=72 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos -wb --train-encoder --wandb-name cr-on-tr-trnenc-005 -lr 0.005"
#oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos --trim -wb --train-encoder --wandb-name cr-on-tr-trnenc-0001 -lr 0.0001"
oarsub -l gpu=1,walltime=8 "source ~/.bashrc; conda activate main; cd ~/work/coref; python src/run.py -d ontonotes -t coref -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 300 -dv cuda --debug --filter-candidates-pos  -wb --train-encoder --wandb-name cr-on-tr-trnenc-0005 -lr 0.0005"