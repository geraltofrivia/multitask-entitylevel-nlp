rm data/parsed/*/*/*
rm -r data/encoded/*

python src/preproc/ontonotes.py -a
python src/preproc/scierc.py -s all
python src/preproc/codicrac.py -a

# For my next trick, if you have ze GPUs you wanna run (For all datasets).
# Replace the -enc with whatever you're currently using
# TODO: replace this if this becomes problematic
python src/run.py -d ontonotes -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d ontonotes -t ner 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d scierc -t ner 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d codicrac-persuasion -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d codicrac-light -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d codicrac-switchboard -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
python src/run.py -d codicrac-ami -t coref 1.0 True -tok bert-base-cased -enc SpanBERT/spanbert-base-cased -e 1000 -dv cuda -lr 0.0005 --lr-schedule gamma 0.80 encode
