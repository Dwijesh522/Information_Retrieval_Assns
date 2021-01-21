import Parser
from os import listdir
import time
import re
import json

class DocumentProcessing:
    def __init__(self, directory_path, indexfile):
        # directory containint files, files
        # containing many documents
        self.directory_path = directory_path

        # store dict, posting_list as
        # indexfile.dict, indexfile.idx
        self.indexfile = indexfile;

        # list of file names in directory_path
        # cold function
        self.files = self.getFiles()

        # stores unique words, unique documents
        self.parser = Parser.Parser()

        # call self.parser.feed() for all documents
        # It will collect all unique words in parser
        ###### HOT FUNCTION
        self.collectWords()

        # dictionary of unique words
        # seen in all documents
        self.dict = dict(zip(self.parser.words, range(len(self.parser.words))))
        
        # stores the posting list
        self.posting_list_parser = Parser.PostingListParser(self.parser.words)

        # given dictionary, and list of all file names
        # posting_list: posting list for each word in dictionary
        # type:         dict(word : list(document_ids))
        # Note:         duplicate document ids in a posting list is
        #               allowed and helps calculate freq of that 
        #               word in that duplicate document.
        ###### HOT FUNCTION
        self.updatePostingList()

        # write dictionary and posting list to disk
        # in binary form(smallest possible size).
        self.dump()

    def getFiles(self):
        """
            given self.directory_path, 
            returns list of file names
                    without prefix path
        """
        return listdir(self.directory_path)
        
    def collectWords(self):
        """
            input:  
                implicite. self.parser() object,
                self.files

            output: 
                implicite. writes to parser.words

            semantics:
                calls parser.feed() method for all
                documents to collect unique words.

            ###### HOT FUNCTION
        """
        for f in self.files:
            with open(self.directory_path + '/' + f, 'r') as file:
                # reading the whole file and removing '\n' and '`' noise
                lots_of_docs = file.read()
                lots_of_docs = re.sub('\n', ' ', lots_of_docs)
                lots_of_docs = re.sub('`', '', lots_of_docs)
            self.parser.appendToWords(lots_of_docs)
        
        # sorting all words for *query
        self.parser.sort()
        del(lots_of_docs)

    def updatePostingList(self):
        """
            input:
                implicite. self.dict, self.files

            output:
                updates self.posting_list_parser.posting_list

            semantics:
                for all words in dictionary, store
                corresponding posting list in self.posting_list_parser

            ###### HOT FUNCTION
        """
        for f in self.files:
            with open(self.directory_path + '/' + f, 'r') as file:
                lots_of_docs = file.read()
                lots_of_docs = re.sub('\n', ' ', lots_of_docs)
                lots_of_docs = re.sub('`', '', lots_of_docs)
            self.posting_list_parser.appendToPostingList(lots_of_docs)
        del(lots_of_docs)

    def dump(self):
        """
            input:
                implicite. self.dict, self.posting_list
            output:
                implicite. Write dict in indexfile.dict
                and write posting_list in indexfile.idx

            semantics:
                write self.dict and self.posting_list
                in binary(s.t. it occupies smallest size)
        """
        json.dump(self.dict, open(self.indexfile+'.dict', 'w'))
        json.dump(self.posting_list_parser.posting_list, open(self.indexfile+'.idx', 'w'))
#        data = json.load( open( "file_name.json" ) )
