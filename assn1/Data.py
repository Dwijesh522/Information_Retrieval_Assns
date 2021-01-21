import Parser
import json
import time

class Data:
    def __init__(self, dictionary_path, posting_list_path):
        # path to compressed dictionary
        self.dictionary_path = dictionary_path

        # path to compressed posting list
        self.posting_list_path = posting_list_path

        # type: dict(word : index)
        self.vocab = json.load(open(dictionary_path))
        
        # type: dict(word : list(document_ids))
        self.posting_list = json.load(open(posting_list_path))

        # set of unique docids seen self.posting_list
        self.docids = set()
        self.fetchUniqueDocids()

    def fetchUniqueDocids(self):
        """
            reads from self.posting_list
            updates set of unique docids seen
            ###### HOT FUNCTION
        """
        self.docids = set([docid for lots_of_docids in self.posting_list.values()\
                                    for docid in lots_of_docids])
