echo "This script can't download CODICRAC/Ontonotes or ACE datasets for you. Please see src/playing-with-data-codicrac.ipynb to see what they should be arranged like."

mkdir data
mkdir data/raw
mkdir data/raw/ontonotes
mkdir data/raw/scierc
mkdir data/raw/codicrac-switchboard
mkdir data/raw/codicrac-light
mkdir data/raw/codicrac-persuasion
mkdir data/raw/codicrac-ami
mkdir data/raw/codicrac-arrau
mkdir data/raw/codicrac-arrau-t91
mkdir data/raw/codicrac-arrau-t93
mkdir data/raw/codicrac-arrau-rst
mkdir data/raw/codicrac-arrau-gnome
mkdir data/raw/codicrac-arrau-pear
mkdir data/raw/ace2005
mkdir data/raw/dwie
mkdir data/parsed
mkdir data/parsed/ontonotes
mkdir data/runs
mkdir data/runs/ne_coref
mkdir models
mkdir models/glove
mkdir models/wordtovec
mkdir models/trained

# Downloading conll formatted ontonotes skeleton files
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/refs/tags/v12.tar.gz -P data/raw/ontonotes
mkdir data/raw/ontonotes/conll-2012
tar -xvf data/raw/ontonotes/v12.tar.gz -C data/raw/ontonotes
mv data/raw/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0 data/raw/ontonotes/conll-2012
mv data/raw/ontonotes/conll-2012/conll-formatted-ontonotes-5.0 data/raw/ontonotes/conll-2012/v5
rm -r data/raw/ontonotes/conll-formatted-ontonotes-5.0-12

# Downloading pre-processed SciERC dataset
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz -P data/raw/scierc
tar -xvf data/raw/scierc/sciERC_processed.tar.gz -C data/raw/scierc
mv data/raw/scierc/processed_data/json/* data/raw/scierc/
rm -r data/raw/scierc/processed_data

# Running scripts to convert .skel files to .conll files
src/preproc/conll-2012/v3/scripts/skeleton2conll.sh -D data/raw/ontonotes/ontonotes-release-5.0/data/files/data data/raw/ontonotes/conll-2012

# Installing dependencies
pip install -r requirements.txt

# Get all submodules running
git submodule update --init --recursive

# Downloading the spacy model
python -m spacy download en_core_web_sm-3.1.0 --direct

# Downloading glove
wget https://nlp.stanford.edu/data/glove.6B.zip -P models/glove
unzip models/glove/glove.6B.zip -d models/glove/

# For DWIE corpora
cd modules/dwie
python src/dwie_download.py
cd ../..
# now move the raw data far from the `modules` dir, into the `raw` dir
mkdir -pv data/raw/dwie
mv modules/dwie/data/annos_with_content/* data/raw/

# For ACE 2005 Corpora
mkdir modules/ace2005-toolkit/ace_2005
tar zxvf data/raw/ace2005/ace_2005_td_v7_LDC2006T06.tgz -C modules/ace2005-toolkit/ace_2005/
mv modules/ace2005-toolkit/ace_2005_td_v7/* modules/ace2005-toolkit/
rm -r modules/ace2005-toolkit/ace_2005_td_v7/
cd modules/ace2005-toolkit/
pip install -r requirements.
chmod +x run.sh
./run.sh en     # Now we need a user prompt. Say 'y' for yes at the time.
cd ../..
mv modules/ace2005-toolkit/cache_data/English/* data/raw/ace2005
./preproc.sh