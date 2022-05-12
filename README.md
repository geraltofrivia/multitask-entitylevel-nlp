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
│   │   ├── codicrac-arrau
│   │   │   └── ARRAU2.0_UA_v3_LDC2021E05.zip
│   │   ├── codicrac-arrau-gnome
│   │   │   └── Gnome_Subset2.CONLL
│   │   ├── codicrac-arrau-pear
│   │   │   └── Pear_Stories.CONLL
│   │   ├── codicrac-arrau-rst
│   │   │   ├── RST_DTreeBank_dev.CONLL
│   │   │   ├── RST_DTreeBank_test.CONLL
│   │   │   └── RST_DTreeBank_train.CONLL
│   │   ├── codicrac-arrau-t91
│   │   │   └── Trains_91.CONLL
│   │   ├── codicrac-arrau-t93
│   │   │   └── Trains_93.CONLL
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
2. Get access to CODI CRAC 2022 datasets (IDK how, again, sorry).
3. Download the `conll-2012-scripts.v3.tar.gz` scripts (find the name in page) from [this page](https://cemantix.org/conll/2012/data.html) and extract them to `src/preproc/`
4. Run `setup.sh` to download, untar and process the conll-2012 skeleton files into conll formatted files. This might take some time.

### Running Experiments

Do not. As of 12-05-2022 the code still does not work. I have spent 5 weeks debugging it but I haven't managed to get it work.
I'm a bad programmer, je sais, mais qu'est-ce que je peux faire donc ?