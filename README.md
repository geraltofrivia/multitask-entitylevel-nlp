## Set Up

### Directory Tree:

```bash
priyansh@priyansh-ele:~/Dev/research/coref/mtl$ tree -L 4
.
├── data
│   ├── linked
│   ├── manual
│   │   ├── ner_ontonotes_tag_dict.json
│   │   ├── ner_scierc_tag_dict.json
│   │   ├── rel_scierc_tag_dict.json
│   │   └── replacements.json
│   ├── parsed
│   │ <...> 
│   ├── raw
│   │   ├── codicrac-ami
│   │   │   └── AMI_dev.CONLLUA
│   │   ├── codicrac-light
│   │   │   ├── light_dev.CONLLUA
│   │   │   ├── light_dev.CONLLUA.zip
│   │   │   └── __MACOSX
│   │   ├── codicrac-persuasion
│   │   │   └── Persuasion_dev.CONLLUA
│   │   ├── codicrac-switchboard
│   │   │   ├── __MACOSX
│   │   │   ├── Switchboard_3_dev.CONLL
│   │   │   └── Switchboard_3_dev.CONLL_LDC2021E05.zip
│   │   ├── ontonotes
│   │   │   ├── conll-2012
│   │   │   ├── ontonotes-release-5.0
│   │   │   ├── ontonotes-release-5.0_LDC2013T19.tgz
│   │   │   └── v12.tar.gz
│   │   ├── ace2005
│   │   │   └──ace_2005_td_v7_LDC2006T06.tgz
│   │   └── scierc
│   │       ├── dev.json
│   │       ├── sciERC_processed.tar.gz
│   │       ├── test.json
│   │       └── train.json
│   └── runs
│       └── ne_coref
│           ├── goldner_all.json
│           ├── goldner_some.json
│           ├── spacyner_all.json
│           └── spacyner_some.json
├── g5k.sh
├── models
│ <...>
├── preproc.sh
├── README.md
├── requirements.txt
├── setup.sh
├── src
│   ├── analysis
│   │   └── ne_coref.py
│   ├── config.py
│   ├── dataiter.py
│   ├── eval.py
│   ├── loops.py
│   ├── models
│   │   ├── autoregressive.py
│   │   ├── embeddings.py
│   │   ├── modules.py
│   │   ├── multitask.py
│   │   ├── _pathfix.py
│   │   └── span_clf.py
│   ├── _pathfix.py
│   ├── playing-with-data-codicrac.ipynb
│   ├── preproc
│   │   ├── codicrac.py
│   │   ├── commons.py
│   │   ├── ontonotes.py
│   │   ├── _pathfix.py
│   │   └── scierc.py
│   ├── run.py
│   ├── utils
│   │   ├── data.py
│   │   ├── exceptions.py
│   │   ├── misc.py
│   │   ├── nlp.py
└── todo.md
...
```

### Gettting multiple datasets

We follow the instructions from [this cemantix page](https://cemantix.org/data/ontonotes.html) but the "scripts" mention there can't be downloaded. Those can be found [here](https://cemantix.org/conll/2012/data.html) but
we'll download these automatically.

1. Get access to `ontonotes-release-5.0` somehow from LDC (I still hate them to make a common dataset propriotary but what can I do). And put it inside `data/raw/ontonotes`. See tree above to figure out the 'right' way
   to arrange ontonotes.
2. Similarly get access to ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06).
3. Get access to CODI CRAC 2022 datasets (IDK how, again, sorry).
4. Download the `conll-2012-scripts.v3.tar.gz` scripts (find the name in page) from [this page](https://cemantix.org/conll/2012/data.html) and extract them to `src/preproc/`
5. Run `setup.sh` to download, untar and process the conll-2012 skeleton files into conll formatted files. This might take some time.
   1. At some point you would be asked for something (ACE dataset) to be divided by sentence level. Enter 'y'.
   2. It will ask for train dev test ratios: enter TODO (tentatively: `0.7 0.15 0.15`)
   3. It will then ask to transform BIO tags or not: enter y (TODO: you sure?)
6. This will also make multple changes including downloading some downloadable stuff, installing dependencies etc.
7. Run `./preproc.sh` to convert RAW datasets into a consistent format (outputs would be saved in `~/data/parsed`).

### Running Experiments

TODO: mention some example configs.