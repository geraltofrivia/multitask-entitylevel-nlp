rm data/parsed/*/*/*

python src/preproc/ontonotes.py -a
python src/preproc/scierc.py -s all
python src/preproc/codicrac.py -a