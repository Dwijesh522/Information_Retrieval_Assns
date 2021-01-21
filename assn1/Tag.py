import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.stem import PorterStemmer
import os
import time
#from nltk.tokenize import RegexpTokenizer

"""
    PERFORMANCE OPTIMIZATION POSSIBILITY
        use nltk tagger: https://pythonprogramming.net/testing-stanford-ner-taggers-for-speed/
"""

def tag_entities(filepath):

    jar = './stanford-ner.jar'
    model = './english.all.3class.distsim.crf.ser.gz'
    
    ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')
    
    tags = ['PERSON', 'LOCATION','ORGANIZATION']
    punc = ['.',',',':',';','"','!','?','%']
    suffix_lookup = {  'P' : ['person'], 'O' : ['organization'], 'L' : ['location'],\
                        'N' : ['person', 'organization', 'location']}
       
    #type: dict(query_id(=int), list[query words(=string)])
    queries = {}
    
    # stemming words
    ps = PorterStemmer()

    with open(filepath) as f:
        data = f.read()
        words_ner = []
        for text in data.split("<desc>"):
            if "<num>" in text:
                # fetching query_count
                query_count = int(text[ text.find("<num>")+len("<num> Number: "):\
                                        text.find('<dom>')].split()[0])
                queries[query_count] = []
                # fetching query
                doc = text[text.find("<title>")+len("<title> Topic:"):]
                tokenized = nltk.sent_tokenize(doc)
                for i in tokenized:
                    wordsList = i.split()
                    words_ner = ner_tagger.tag(wordsList)
                    for j in words_ner:
                        ## handling tag specific query
                        qtype_query = j[0].split(':')
                        suffix = suffix_lookup.get(qtype_query[0])
                        # no tag found 
                        if (suffix == None):
                            if j[1] in tags:
                                queries[query_count] += [j[0].lower()+j[1].lower()]
                            else:
                                queries[query_count] += [ps.stem(j[0].lower())]
                        # tag found
                        else:
                            queries[query_count] += [qtype_query[1].lower() + x for x in suffix]
#            if (query_count == 51): break
    return queries

#class Tag:
#    def __init__(self, query_path):
#        # path to query file.
#        self.query_path = query_path
#
#        # queries is a list of strings, where
#        # each string is a query.
#        self.queries = self.fetchQueries()
#
#        # tagged queries
#        self.queries = self.tagQueries()
#
#    def fetchQueries(self):
#        """
#            reads from self.query_path
#            returns list of strings where
#                    each string is query
#        """
#        queries = []
#        return queries
#
#    def tagQueries(self):
#        """
#            reads from self.queries
#            returns list of strings where each 
#                    string is tagged a query.
#        """
#        return self.queries
