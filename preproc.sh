rm data/parsed/*/*/*

python src/preproc/ontonotes.py -a
python src/preproc/scierc.py -s all
python src/preproc/codicrac.py -a

# For my next trick, if you have ze GPUs you wanna run (For all datasets).
# Replace the -enc with whatever you're currently using
# python src/run.py -d ontonotes -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode