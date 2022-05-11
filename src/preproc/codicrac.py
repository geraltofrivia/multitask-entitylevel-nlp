"""
    Here we aim to parse every CODI CRAC raw dataset.
    This is to say that we start with the raw files and turn them into Document (utils/data) instances.
    One object corresponds to one document.
"""
import re
import json
import click
import spacy
import jsonlines
from spacy import tokens
from pathlib import Path
from typing import Iterable, Union, List, Dict, Optional

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import LOCATIONS as LOC, KNOWN_SPLITS
from preproc.commons import GenericParser
from dataiter import DocumentReader
from utils.nlp import to_toks, NullTokenizer
from utils.data import Document, NamedEntities, TypedRelations, Clusters
from utils.misc import NestedAnnotationBlock, NestedAnnotationBlockStack


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

    def __init__(self, raw_dir: Path, ignore_empty_documents: bool = False):

        # Sanity add suffixes as per raw_dir
        self.dataset = raw_dir.name
        self.crac_src = self.dataset.split('-')[1]
        possible_suffixes = {
            'codicrac-ami': ['AMI_dev.CONLLUA'],
            'codicrac-persuasion': ['Persuasion_dev.CONLLUA'],
            'codicrac-light': ['light_dev.CONLLUA'],
            'codicrac-switchboard': ['Switchboard_3_dev.CONLL'],
            'codicrac-arrau-t91': ['Trains_91.CONLL'],
            'codicrac-arrau-t93': ['Trains_93.CONLL'],
            'codicrac-arrau-gst': [
                'RST_DTreeBank_dev.CONLL',
                'RST_DTreeBank_test.CONLL'
                'RST_DTreeBank_train.CONLL'
            ],
            'codicrac-arrau-pear': ['Pear_Stories.CONLL'],
            'codicrac-arrau-gnome': ['Gnome_Subset2.CONLL'],
        }
        suffix = possible_suffixes[self.dataset]

        super().__init__(raw_dir=raw_dir, suffixes=suffix, ignore_empty_documents=ignore_empty_documents)

        self.write_dir = LOC.parsed / self.dataset
        self.write_dir.mkdir(parents=True, exist_ok=True)

        # Null tokenizer is needed when text is pre-tokenized
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = NullTokenizer(self.nlp.vocab)

        # noinspection RegExpRedundantEscape
        self.re_nested_annotation_blocks = re.compile(r'\([^\)\(\n]*\)?|\)')

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

    def parse(self, split_nm: Union[Path, str]) -> List[Document]:
        """ where actual preproc happens"""

        outputs: List[Document] = []
        filedir: Path = self.dir / split_nm

        assert filedir.exists()

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
        known_speakers: Dict[str, int] = {}
        current_speaker = 10 if self.crac_src == 'light' else -1  # light begins with 'setting' which is new speaker

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

                if self.crac_src in ['light', 'persuasion', 'ami'] and 'speaker' in tokens:
                    # A special metadata line: speaker information
                    # E.g. "# speaker = orc"
                    current_speaker = known_speakers.setdefault(tokens[-1], len(known_speakers))

                if 'newdoc' in tokens:
                    # We have started a document. Dump the old one and store the key for the new one

                    if raw_document:
                        documents[docname] = raw_document
                        documents_[docname] = document_
                        documents_speakers[docname] = document_speakers

                    raw_document, document_, document_speakers = [], [], []
                    docname = tokens[-1]
                    current_speaker = 10

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

        # Strip these annotation blocks of all metadata, and only keep spans and markable names
        markables: Dict[str, Dict[str, NestedAnnotationBlock]] = {docname: {block.tag: block for block in blocks}
                                                                  for docname, blocks in markables_.items()}

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
            # noinspection PyTypeChecker
            doc = self.nlp(to_toks(doc_text))
            doc_pos = self.get_pos_tags(doc)

            # Get the named entities prepped
            doc_named_entities = named_entities[docname]
            ner_spans = [[x.start, x.end] for x in doc_named_entities]
            ner_tags = [x.metadata for x in doc_named_entities]
            ner_words = [x.words for x in doc_named_entities]
            ner = NamedEntities(spans=ner_spans, tags=ner_tags, words=ner_words)

            # Get the coref clusters prepped
            doc_clusters_spans = [[[span.start, span.end] for span in cluster]
                                  for cluster in list(documents_clusters[docname].values())]
            coref = Clusters(spans=doc_clusters_spans)

            # Get the bridging things prepped as well
            ...

        return outputs


if __name__ == '__main__':
    parser = CODICRACParser(LOC.cc_persuasion)
    parser.run()
