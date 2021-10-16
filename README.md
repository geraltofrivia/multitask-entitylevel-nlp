## Set Up

### Getting OntoNotes done right.

We follow the instructions from [this cemantix page](https://cemantix.org/data/ontonotes.html) but the "scripts" mention there can't be downloaded. Those can be found [here](https://cemantix.org/conll/2012/data.html) but we'll download these automatically.

1. Get access to `ontonotes-release-5.0` somehow from LDC (I still hate them to make a common dataset propriotary but what can I do). And put it inside `data/raw/ontonotes`
It should look like:
```bash
.
├── data
│     └── raw
│         └── ontonotes
│             ├── ontonotes-release-5.0
│             │     ├── data
│             │     ├── docs
│             │     ├── index.html
│             │     └── tools
│             └── ontonotes-release-5.0_LDC2013T19.tgz
│     └── README.md
└── setup.sh
...
```
2. Download the `conll-2012-scripts.v3.tar.gz` scripts (find the name in page) from [this page](https://cemantix.org/conll/2012/data.html) and extract them to `src/preproc/`
3. Run `setup.sh` to download, untar and process the conll-2012 skeleton files into conll formatted files. This might take some time.
