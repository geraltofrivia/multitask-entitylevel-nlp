rm data/parsed/*/*/*
rm -r data/encoded/*

python src/preproc/ontonotes.py -a
python src/preproc/scierc.py -s all
python src/preproc/codicrac.py -a
python src/preproc/dwie.py

# For my next trick, if you have ze GPUs you wanna run (For all datasets).
# Replace the -enc with whatever you're currently using
# TODO: replace this if this becomes problematic
