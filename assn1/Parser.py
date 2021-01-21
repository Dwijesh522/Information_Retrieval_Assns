from html.parser import HTMLParser
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

"""
    ASSUMPTIONS:
        1) Tags <PERSON>, <ORGANIZATION>, and <LOCATION> enclose only one word
"""

"""
    Opportunities for reducing time:
        1) Reducing dictionary size
"""

"""
    Tags of interest:   <DOC>, </DOC>, new doc will always start with this
                        <PERSON>, </PERSON>,
                        <ORGANIZATION>, </ORGANIZATION>,
                        <LOCATION>, </LOCATION>
                        <DOCNO>, </DOCNO>
                        <TEXT>, </TEXT> anything b/w them is important
                        <NOTE>, </NOTE>
"""

"""
    parses HTML documents.

    How to use:
        parser = Parser()
        parser.feed('<doc> My Doc <\doc>')

    ###### HOT CLASS

    Reference: https://docs.python.org/3/library/html.parser.html
"""
class Parser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.words = set()
        self.state = ["none"]
        
        # frequently used variables
        self.punctuations = string.punctuation
        self.pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        self.ps = PorterStemmer()

    def handle_starttag(self, tag, attrs):
        self.state += [tag]

    def handle_endtag(self, tag):
        self.state.pop()

    def handle_data(self, data):
        curr_state = self.state[len(self.state)-1]
        if  (curr_state == 'person') | \
            (curr_state == 'organization') | \
            (curr_state == 'location'):
            # preprocessing data
            data = re.sub(' ', '', data)
            data = data.lower()

            self.words.add(data + curr_state)
        elif (curr_state == 'text'):
            # no preprocessing is done as of now
            # because now a days recent browsers
            # does not preprocess.

            # removing punctuations
            data = re.sub('['+self.punctuations+']', ' ', data)
            data = data.lower()

            self.words.update(set([self.ps.stem(w) for w in data.split()]))  # splits wrt space

    def appendToWords(self, docs):
        """
            input:
                docs:   string. representing all 
                        docs present in a file
            output:
                void

            semantics:  keep appending unique words
                        to the set self.words
        """
        # stop word removal
        docs = self.pattern.sub('', docs)
        self.feed(docs)

    def sort(self):
        """
            sort all words stored in the set
        """
        self.words = set(sorted(self.words))

"""
###### HOT CLASS
"""
class PostingListParser(HTMLParser):
    def __init__(self, words):
        HTMLParser.__init__(self)
        # type(word) = set(string)
        # type: dict(word, list(doc_id))
        # initializing the posting_list 
        # with empty list for each word
        self.posting_list = dict(zip(words, [[] for _ in range(len(words))] ))
        self.state = ["none"]
        self.curr_docid = "none"
        
        # frequently used variables
        self.punctuations = string.punctuation
        self.pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        self.ps = PorterStemmer()
    
    def handle_starttag(self, tag, attrs):
        self.state += [tag]

    def handle_endtag(self, tag):
        self.state.pop()

    def handle_data(self, data):
        curr_state = self.state[len(self.state)-1]
        if (curr_state == 'docno'):
            # preprocessing data
            data = re.sub(' ', '', data)
            self.curr_docid = data
        elif    (curr_state == 'person') | \
                (curr_state == 'organization') | \
                (curr_state == 'location'):
            # preprocessing data
            data = re.sub(' ', '', data)
            data = data.lower()

            self.posting_list[data + curr_state].append(self.curr_docid)
        elif (curr_state == 'text'):
            # removing punctuations
            data = re.sub('['+self.punctuations+']', ' ', data)
            data = data.lower()
            data = [self.ps.stem(w) for w in data.split()]

            for word in data:
                self.posting_list[word].append(self.curr_docid)
    def appendToPostingList(self, docs):
        """
            input:
                docs:   string. representing all 
                        docs present in a file
            output:
                void

            semantics:  keep appending unique words
                        to the set self.words
        """
        # removing stop words
        docs = self.pattern.sub('', docs)
        self.feed(docs)
