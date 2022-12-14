"""
    Here we aim to parse every CODI CRAC raw dataset.
    This is to say that we start with the raw files and turn them into Document (utils/data) instances.
    One object corresponds to one document.
"""
import json
import re
from pathlib import Path
from typing import Union, List, Dict, Optional

import click

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC
from preproc.commons import GenericParser
from dataiter import DocumentReader
from utils.nlp import to_toks
from utils.data import Document, NamedEntities, BridgingAnaphors, Clusters, TypedRelations
from utils.misc import NestedAnnotationBlock, NestedAnnotationBlockStack

CODICRAC_SBD_TOKEN = '@$$!!~~!!$$@'  # Just a token that you will NOT see in actual data.


class CODICRACParser(GenericParser):
    """
        The different raw directories (and correspondingly, different CODICRACParser instances) can be found with
            - LOC.cc_switchboard
            - LOC.cc_light
            - LOC.cc_persuasion
            - LOC.cc_ami
            - LOC.cc_arrau_t91
            - LOC.cc_arrau_t93
            - LOC.cc_arrau_rst
            - LOC.cc_arrau_gnome
            - LOC.cc_arrau_pear

        For all of these, you want to invoke the parser like:
            parser = CODICRACParser(raw_dir = LOC.cc_arrau_pear)
            parser.run()
    """

    def __init__(
            self,
            raw_dir: Path,
            write_dir: Path = LOC.parsed,
            ignore_empty_documents: bool = False):

        # Sanity add suffixes as per raw_dir
        # NOTE: in possible suffixes, order matters. So **test set should be the last one**.
        self.dataset = raw_dir.name
        self.crac_src = self.dataset.split('-')[1]
        possible_suffixes = {
            'codicrac-ami': ['2022_AMI_train_v0.CONLLUA', '2022_AMI_dev_v0.CONLLUA'],
            'codicrac-persuasion': ['Persuasion_train.2022.CONLLUA', 'Persuasion_dev.2022.CONLLUA'],
            'codicrac-light': ['light_train.2022.CONLLUA', 'light_dev.2022.CONLLUA'],
            'codicrac-switchboard': ['Switchboard_train.2022.CONLLUA', 'Switchboard_dev.2022.CONLLUA'],
            'codicrac-arrau-t91': ['Trains_91.CONLL'],
            'codicrac-arrau-t93': ['Trains_93.CONLL'],
            'codicrac-arrau-rst': [
                'RST_DTreeBank_train.CONLL',
                'RST_DTreeBank_dev.CONLL',
                'RST_DTreeBank_test.CONLL',
            ],
            'codicrac-arrau-pear': ['Pear_Stories.CONLL'],
            'codicrac-arrau-gnome': ['Gnome_Subset2.CONLL'],
        }
        suffix = possible_suffixes[self.dataset]

        super().__init__(
            raw_dir=raw_dir,
            suffixes=suffix,
            write_dir=write_dir,
            ignore_empty_documents=ignore_empty_documents)

        self.write_dir = LOC.parsed / self.dataset
        self.write_dir.mkdir(parents=True, exist_ok=True)

        # Null tokenizer is needed when text is pre-tokenized
        # self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.tokenizer = self.nlp.tokenizer.tokens_from_list
        # self.nlp.tokenizer = PreTokenizedTokenizer(self.nlp.vocab)
        # self.nlp.add_pipe("codicrac-sbd", first=True)

        # noinspection RegExpRedundantEscape
        self.re_nested_annotation_blocks = re.compile(r'\([^\)\(\n]*\)?|\)')

    # @staticmethod
    # @Language.component("codicrac-sbd")
    # def codicrac_sbd(doc: tokens.Doc) -> tokens.Doc:
    #     indexes = []
    #     for token in doc[:-1]:
    #         if token.text == CODICRAC_SBD_TOKEN:
    #             doc[token.i + 1].is_sent_start = True
    #             indexes.append(token.i)
    #
    #     np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    #     np_array = np.delete(np_array, indexes, axis=0)
    #     doc2 = tokens.Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    #     doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    #     return doc2

    def run(self):
        """ overwriting it since we don't need suffixes UNLESS we're dealing with RST"""

        for split in self.suffixes:
            # First, clear out all the previously processed things from the disk
            self.delete_preprocessed_files(split)
            outputs = self.parse(split)

            # # Dump them to disk
            # if self.dataset != 'codicrac-arrau-rst':
            #     self.write_to_disk(suffix=None, instances=outputs)
            # else:
            # We are dealing with RST. Which split?
            if '_dev' in split:
                self.write_to_disk(suffix='dev', instances=outputs)
            elif '_test' in split:
                self.write_to_disk(suffix='test', instances=outputs)
            elif '_train' in split:
                self.write_to_disk(suffix='train', instances=outputs)
            else:
                raise ValueError(f"RST filename: {split} is unknown.")

        # Dump speaker IDs to disk as well.
        self.create_label_dict()

    @staticmethod
    def _parse_annotation_(annotation: str) -> Dict[str, str]:
        """
        INPUT
            (EntityID=1-Pseudo|MarkableID=markable_469|Min=16,17|SemType=quantifier)
        OUTPUT
            {'EntityID': '1-Pseudo', 'MarkableID': 'markable_469', 'Min': '16,17', 'SemType': 'quantifier'}
        """

        if annotation[0] == '(':
            annotation = annotation[1:]

        if annotation[-1] == ')':
            annotation = annotation[:-1]

        parsed: Dict[str, str] = {}
        for key_value in annotation.split('|'):
            k = key_value.split('=')[0]
            v = key_value.split('=')[1]
            parsed[k] = v

        return parsed

    def get_markables(self, documents: Dict[str, List[List[str]]],
                      documents_: Dict[str, List[List[Dict[str, str]]]],
                      annotation_key: str, tag_name: str,
                      ignore_pseudo: bool, meta_name: Optional[str] = None) \
            -> Dict[str, List[NestedAnnotationBlock]]:
        """
            :param documents: A dict of list of list of tokens. Key is doc name
            :param documents_: same as above but instead of token str, we have a dict of all annotations from raw file
            :param annotation_key: which annotation key are we extracting for. E.g. IDENTITY (for markables)
            :param tag_name: we might want to have some text information saved up. which field.
                e.g. in  'MarkableID' in `(EntityID=1-Pseudo|MarkableID=markable_469|Min=16,17|SemType=quantifier)`
            :param meta_name: if given also store some more text information from the specified field. same as above.
            :param ignore_pseudo: don't include it if Pseudo is present in the annotation.
                e.g. Pseudo in EntityID in  `(EntityID=1-Pseudo|MarkableID=markable_469|Min=16,17|SemType=quantifier)`
            :return { docname - 1:
                        { markable tag name - 1: [span start int, span end int],
                          markable tag name - 2: [span start int, span end int],
                          ...
                        },
                      docname - 2:
                        { markable tag name - 3: [span start int, span end int],
                          markable tag name - 4: [span start int, span end int],
                          ...
                        }
                      ...
                    }

           # Markable Detection

               **START**
               an annotation like `(EntityID=1-Pseudo|MarkableID=markable_469|Min=16,17|SemType=quantifier`
               in the 'IDENTITY' field (for e.g.) of a token marks the start of a markable.

               **END**
               Its end is found by the bracket close.

               We assume they can be nested.

           # Algorithm
                We use AnnotationBlock and AnnotationBlock stack used initially in
                    `src/preproc/base class OntonotesBasedParser def _get_pos_spans_from_ontonotes_onf_`

                `( ....`
                Every time we find an open bracket as specified above,
                    we create a new AnnotationBlock and append it to an AnnotationBlockStack

                We also append `1` to all open (i.e. in the stack) AnnotationBlock's 'open' counter in the stack.

                `)`
                We append `1` to all open (i.e. in the stack) AnnotationBlock's 'closed' counter in the stack.
                When open and closed equate, i.e. this block is closed. \
                We pop it from stack and save it in a permanent-ish list.

                Else
                On each subsequent word, we append the word and add its ID to all elements of the AnnotationBlockStack.
        """
        markables_: Dict[str, List[NestedAnnotationBlock]] = {}
        for docname, document, document_ in zip(documents.keys(), documents.values(), documents_.values()):

            stack: NestedAnnotationBlockStack = NestedAnnotationBlockStack()
            finished_blocks: List[NestedAnnotationBlock] = []
            cumulative_token_id_till_sent_start: int = 0

            for sentence_id, (sentence, sentence_) in enumerate(zip(document, document_)):
                for token_id, (token, token_) in enumerate(zip(sentence, sentence_)):

                    assert token == token_['FORM'], f"token: `{token}` and token_: `{token_['FORM']}` do not match!"

                    annotation: str = token_[annotation_key]

                    # In all cases
                    # Add word to all active blocks
                    stack.register_word(token)

                    # Find all annotation blocks
                    # E.g. of blocks
                    # ( ..... -> [(....]
                    # (...(...)(.... -> [(... | (...) | (...]
                    # ) -> [)]
                    # )) -> [) | )]
                    # ( .... ) -> [ (...) ]
                    # Content inside the block ... -> EntityID=5-Pseudo....
                    # Each of these blocks will be processed and replaced in the actual str `annotation` with `_`
                    for annotation_block in self.re_nested_annotation_blocks.finditer(annotation):
                        _span_b, _span_e = annotation_block.span()
                        annotation_block_: str = annotation_block.group()

                        # Case A: annotation has an open bracket
                        if '(' in annotation_block_:

                            if ignore_pseudo and 'pseudo' in annotation_block_.split('|')[0].lower():
                                # Ignore all pseudo ones (they are singletons)
                                continue

                            # Update active blocks with an open bracket
                            stack.add(1)

                            # Parse contents within a block, so we can note down markable IDs and such
                            parsed_annotation_block: Dict[str, str] = self._parse_annotation_(annotation_block_)

                            new_instance: NestedAnnotationBlock = NestedAnnotationBlock(
                                start=cumulative_token_id_till_sent_start + token_id,
                                words=[token],
                                open_inside=1,
                                tag=parsed_annotation_block[tag_name]
                            )

                            if meta_name:
                                new_instance.metadata = parsed_annotation_block.get(meta_name, '')

                            stack.append(new_instance)

                        # Case B: (not mutually exclusive): annotation has a closed bracket
                        if ')' in annotation_block_:

                            assert annotation_block_.count(')') == 1, "There are more than one `(` in this block!"

                            # Update active blocks with closed brackets. Catch all those that are _finished_.
                            _finished_blocks: list = stack.sub(1, span=cumulative_token_id_till_sent_start + token_id)

                            if _finished_blocks:
                                finished_blocks += _finished_blocks

                        # UNCONDITIONALLY replace this span with '_' in the main string
                        annotation = annotation[:_span_b] + '_' * (_span_e - _span_b) + annotation[_span_e:]

                # Done Iterating over this sentence. Note the number of tokens in it.
                cumulative_token_id_till_sent_start += len(sentence)

            # Done iterating over all sentences of this document

            # Check if stack is empty
            assert len(stack.blocks) == 0

            # Add the _finished_ ones to the main dict
            markables_[docname] = finished_blocks

        # Sanity check the markables
        for docname, doc_markables in markables_.items():
            doc_toks: List[str] = to_toks(documents[docname])
            for markable in doc_markables:
                assert doc_toks[markable.start: markable.end] == markable.words

        return markables_

    @staticmethod
    def find_antecedents(
            markables: Dict[str, List[NestedAnnotationBlock]],
            anaphors: Dict[str, List[NestedAnnotationBlock]]
    ) -> Dict[str, List[NestedAnnotationBlock]]:

        outputs = {}

        for docname in markables.keys():

            outputs[docname] = []
            # Turn List[NestedAnnotationBlock] to Dict[str, NestedAnnotationBlock] i.e. markable ID to block dict
            doc_markables = {block.tag: block for block in markables[docname]}

            for anaphor in anaphors[docname]:
                antecedent_markable_id = anaphor.metadata
                try:
                    antecedent_markable = doc_markables[antecedent_markable_id]
                except KeyError:
                    # No markable antecedent was found. Investigate why
                    print('No markable antecedent was found for bridging. Figure out why. ')
                    raise KeyError
                outputs[docname].append(antecedent_markable)

        return outputs

    def parse(self, split_nm: Union[Path, str]) -> List[Document]:
        """ where actual preproc happens"""

        _is_test_split = '_test' in split_nm

        outputs: List[Document] = []
        filedir: Path = self.dir / split_nm

        if not filedir.exists():
            raise FileNotFoundError(f"File {filedir} not found!")

        # Lets read the file
        with filedir.open('r') as f:
            raw = f.readlines()

        """
            Init some variables we'll use in the doc.

            sentence is a list of tokens in one sentence
            document is a list of sentences in one document
            documents is a docname: doc sentences dict for each document in the file

            their `var_name`_ variants replace the token str with a dict containing all metadata for this token.
        """
        documents: Dict[str, List[List[str]]] = {}
        documents_: Dict[str, List[List[Dict[str, str]]]] = {}
        documents_speakers: Dict[str, List[int]] = {}

        # Init the first speaker in the case of light with an actual value
        if self.crac_src == 'light':
            # light begins with 'setting' which is is a different speaker than the rest
            try:
                current_speaker = self._speaker_vocab_['init_speaker']
            except KeyError:
                if _is_test_split:
                    raise KeyError(f"Speaker vocab does not have an instance of init speaker. Is it empty?\n"
                                   f"\n{self._speaker_vocab_}\n"
                                   f"If so, you might have passed the test split before other splits."
                                   f"\nRectify that in possible_suffixes var in __init__ of CODICRACParser class."
                                   f"Test splits should always be at the end."
                                   f"If instead, your dataset does not have any other split but the test one..."
                                   f"\n\t well, contact the repo maintainer I guess.")
                else:
                    self._speaker_vocab_['init_speaker'] = len(self._speaker_vocab_)
                    current_speaker = self._speaker_vocab_['init_speaker']
        else:
            current_speaker = -1

        raw_document: List[List[str]] = []
        document_speakers: List[int] = []
        document_: List[List[Dict[str, str]]] = []

        sentence: List[str] = []
        sentence_: List[Dict[str, str]] = []

        docname: str = ''

        """
            Parse the file to fill the documents and documents_ var.
        """
        # Keys to token annotations are mentioned in the first line, sort of like an csv file.
        keys: List[str] = raw[0].strip().split('=')[1].split()
        last_state: int = -1

        # Iterate after the first line.
        for i, line in enumerate(raw[1:]):

            # Strip and split the line into its tokens
            tokens = line.strip().split()

            if not tokens:
                # New sentence line.
                # Example: '\n', i.e. an empty line

                assert sentence, "Expected sentence to have some tokens. It is empty"

                raw_document.append(sentence)
                document_.append(sentence_)
                document_speakers.append(current_speaker)

                sentence, sentence_ = [], []

                last_state = 0

            elif tokens[0] == '#':
                # Meta data line.
                # Example: '# sent_id = D93_9_1-1\n'

                if self.crac_src in ['light', 'persuasion', 'ami', 'switchboard'] and 'speaker' in tokens:
                    # A special metadata line: speaker information
                    # E.g. "# speaker = orc"
                    current_speaker = self._speaker_vocab_.setdefault(tokens[-1], len(self._speaker_vocab_))

                if 'newdoc' in tokens:
                    # We have started a document. Dump the old one and store the key for the new one

                    if raw_document:
                        documents[docname] = raw_document
                        documents_[docname] = document_
                        documents_speakers[docname] = document_speakers

                    raw_document, document_, document_speakers = [], [], []
                    docname = tokens[-1]

                    # Init the first speaker in the case of light with an actual value
                    if self.crac_src == 'light':
                        # light begins with 'setting' which is is a different speaker than the rest
                        try:
                            current_speaker = self._speaker_vocab_['init_speaker']
                        except KeyError:
                            if _is_test_split:
                                raise KeyError(
                                    f"Speaker vocab does not have an instance of init speaker. Is it empty?\n"
                                    f"\n{self._speaker_vocab_}\n"
                                    f"If so, you might have passed the test split before other splits."
                                    f"\nRectify that in possible_suffixes var in __init__ of CODICRACParser class."
                                    f"Test splits should always be at the end."
                                    f"If instead, your dataset does not have any other split but the test one..."
                                    f"\n\t well, contact the repo maintainer I guess.")
                            else:
                                self._speaker_vocab_['init_speaker'] = len(self._speaker_vocab_)
                                current_speaker = self._speaker_vocab_['init_speaker']
                    else:
                        current_speaker = -1

                last_state = 1

            elif tokens[0].isdigit():
                # Actual word.
                # Example: '3     hello  _  _  _  _  _  _  _  _  _  _  _  _  \n'
                # It will have either 14 or 15 things.

                assert len(keys) - 1 <= len(tokens) <= len(keys), f"Line {i} has {len(tokens)} annotations."

                if len(tokens) == len(keys) - 1:
                    tokens.append('-')

                parsed = {k: v for k, v in zip(keys, tokens)}
                parsed['RAW'] = line

                sentence.append(parsed['FORM'])
                sentence_.append(parsed)

                last_state = 2

        # Finally check the last state and append things based on that
        if not last_state == 2:
            raise ValueError("The first CRAC parsing loop ended unexpectedly")

        # Assuming we last added a word to the sent.
        # #### Time to put that sent in the doc, the last doc in the global list of docs.
        assert sentence, "Expected sentence to have some tokens. It is empty"

        raw_document.append(sentence)
        document_.append(sentence_)
        document_speakers.append(current_speaker)

        # The output of this run is a slightly parsed collection of information here.
        documents[docname] = raw_document
        documents_[docname] = document_
        documents_speakers[docname] = document_speakers

        """
            Get all noun phrases (which may be antecedents for anaphors)
            # TODO: check if we wanna keep pseudo here or not
        """
        markables_: Dict[str, List[NestedAnnotationBlock]] = self.get_markables(documents=documents,
                                                                                documents_=documents_,
                                                                                annotation_key='IDENTITY',
                                                                                tag_name='MarkableID',
                                                                                meta_name='EntityID',
                                                                                ignore_pseudo=False)

        # Use the meta name (entity ID) to get all coref clusters (they have the same entityID, in the IDENTITY column
        # Try also to group the markables_ with the meta value (EntityID which repr the cluster)
        documents_clusters = {docname: {} for docname in markables_.keys()}
        for docname, doc_markables in markables_.items():
            for markable in doc_markables:
                documents_clusters[docname][markable.metadata] = \
                    documents_clusters[docname].get(markable.metadata, []) + [markable]

        # Get all bridging annotations (along with the markables to which they refer
        bridging_anaphors: Dict[str, List[NestedAnnotationBlock]] = self.get_markables(documents=documents,
                                                                                       documents_=documents_,
                                                                                       annotation_key='BRIDGING',
                                                                                       tag_name='MarkableID',
                                                                                       meta_name='MentionAnchor',
                                                                                       ignore_pseudo=False)
        # Corresponding to each, find an antecedent
        bridging_antecedents = self.find_antecedents(markables_, bridging_anaphors)

        # The REFERENCE column has NER info.
        named_entities: Dict[str, List[NestedAnnotationBlock]] = self.get_markables(documents=documents,
                                                                                    documents_=documents_,
                                                                                    annotation_key='NOM_SEM',
                                                                                    tag_name='MarkableID',
                                                                                    meta_name='Entity_Type',
                                                                                    ignore_pseudo=False)

        # Finally use this collected information to create Document (utils/data) object
        for docname, raw_document, document_ in zip(documents.keys(), documents.values(), documents_.values()):
            doc_text = raw_document

            # Sentence Boundaries don't work like this. So we need to break boundaries based on a custom string
            #   that we attach at the end of every sentence.

            # noinspection PyTypeChecker
            # spacy_doc = self._get_spacy_doc_(doc_text)
            spacy_doc = self.nlp(doc_text)
            doc_pos = self.get_pos_tags(spacy_doc)
            doc_speakers = documents_speakers[docname]

            # Get the named entities prepped
            doc_named_entities = named_entities[docname]
            ner_spans = [[x.start, x.end] for x in doc_named_entities]
            ner_tags = [[x.metadata] for x in doc_named_entities]
            ner_words = [x.words for x in doc_named_entities]
            ner = NamedEntities(spans=ner_spans, tags=ner_tags, words=ner_words)

            # Get the coref clusters prepped
            doc_clusters_spans = [[[span.start, span.end] for span in cluster]
                                  for cluster in list(documents_clusters[docname].values())]
            coref = Clusters(spans=doc_clusters_spans)

            # Get the bridging things prepped as well
            doc_bridging_anaphors = [(block.start, block.end) for block in bridging_anaphors[docname]]
            doc_bridging_antecedents = [(block.start, block.end) for block in bridging_antecedents[docname]]
            bridging_spans = [[ante, ana] for ante, ana in zip(doc_bridging_antecedents, doc_bridging_anaphors)]
            bridging_words = [(ante.words, ana.words) for ante, ana in
                              zip(bridging_anaphors[docname], bridging_antecedents[docname])]
            bridging = BridgingAnaphors(spans=bridging_spans, words=bridging_words)
            genre = self.dataset

            # Make the object
            document = Document(
                document=doc_text,
                pos=doc_pos,
                docname=docname,
                coref=coref,
                ner=ner,
                rel=TypedRelations.new(),
                bridging=bridging,
                speakers=doc_speakers,
                genre=genre
            )

            # Now to finalise the instance
            document = self._finalise_instance_(document, spacy_doc=spacy_doc)

            outputs.append(document)

        return outputs

    def create_label_dict(self):
        """ We assume that every split is processed at the same time.
            So now we go out and create a label dict for speakers as well. """

        relevant_splits = ['dev', 'train']
        pos_labels = set()
        for split in relevant_splits:
            reader = DocumentReader(self.dataset, split)
            for doc in reader:
                pos_labels = pos_labels.union(set(to_toks(doc.pos)))
        pos_labels = list(pos_labels)

        write_dir = LOC.manual / f'speaker_{self.dataset}_tag_dict.json'
        with write_dir.open('w+', encoding='utf8') as f:
            json.dump(self._speaker_vocab_, f)
            print(f"Wrote a dict of {len(self._speaker_vocab_)} to {str(write_dir)}")

        write_dir = LOC.manual / f"pos_{self.dataset}_tag_dict.json"
        with write_dir.open('w+', encoding='utf8') as f:
            pos_labels = {tag: i for i, tag in enumerate(pos_labels)}
            json.dump(pos_labels, f)
            print(f"Wrote a dict of {len(pos_labels)} items to {write_dir}")

        self.create_genre_label_dict()

@click.command()
@click.option("--dataset", "-d", type=str, help="The name of the dataset like 'persuasion', 'light', 'arrau' etc.")
@click.option("--suffix", "-s", type=str, default=None,
              help="The name of (LOC.manual / 'pos_ontonotes_tag_dict.json')the suffix (for Arrau) like 't91', 't93', 'gnome', 'pear', or 'rst'.")
@click.option("--run_all", "-a", is_flag=True, help="If flag is given, we process every CODICRAC split.")
def run(dataset: str, suffix: str, run_all: bool):
    if run_all:

        filenotfound: List[Path] = []
        all_sources = [LOC.cc_persuasion, LOC.cc_ami, LOC.cc_light, LOC.cc_switchboard, LOC.cc_arrau_t91,
                       LOC.cc_arrau_t93, LOC.cc_arrau_pear, LOC.cc_arrau_rst, LOC.cc_arrau_gnome]

        for source in all_sources:
            try:
                parser = CODICRACParser(source)
                parser.run()
            except FileNotFoundError:
                filenotfound.append(source)

        print("-----------------------------------------------------------------------")
        print("--             Did not find raw files for these datasets             --")
        for source in filenotfound:
            print(f"--   - {str(source):61s} --")
        print("-----------------------------------------------------------------------")
    else:
        if suffix:
            parser = CODICRACParser(LOC[f"cc_{dataset}_{suffix}"])
        else:
            parser = CODICRACParser(LOC[f"cc_{dataset}"])

        parser.run()


if __name__ == '__main__':
    run()
