import Data
import Tag
from collections import Counter
import math
import time

class TfIdf:
    def __init__(self, dictionary_path, posting_list_path, query_path, cutoff, result_file):
        # bound on number of documents to retrieve
        self.cutoff = cutoff

        # dump output to result_file
        self.result_file = result_file

        # type: dict(query_id(int), list(words(=string))) 
        start = time.time()
        self.queries = Tag.tag_entities(query_path)
        print('\n{} seconds query tagging\n'.format(time.time() - start))

        # stores vocab, posting_list, tokenized query, set(docids)
        start = time.time()
        self.data = Data.Data(dictionary_path, posting_list_path)
        print('\n{} seconds reading vocab, posting_list, docids\n'.format(time.time()-start))

        # inverse document freq of each word in vocab
        # type: dict(word : idf)
        start = time.time()
        self.idf = self.evalInvDocFreq()
        print('\n{} seconds for inverse doc freq\n'.format(time.time() - start))

        # sparse tfidf vector for all documents
        # type: dict(doc_id, dict(word, tfidf))
        start = time.time()
        self.sparse_tfidf_document = self.sparseTfIdfDocument()
        print("\n{} seconds for tfidf for documents\n".format(time.time()-start))

#        # sparse tfidf vector for all queries
#        # type: dict(query_id, dict(word, tfidf))
#        start = time.time()
#        self.sparse_tfidf_queries = self.sparseTfIdfQueries()
#        print('\n{} seconds for tfidf for queries\n'.format(time.time()-start))
#        return
#
#        # type: dict(id, tfidf vector mod)
#        self.tfidf_mod_documents = self.tfidfVectorMod('doc')
#        self.tfidf_mod_queries   = self.tfidfVectorMod('query')

        del(self.idf)
        del(self.data)

        # for each query, fetch relevent docs based on
        # similarity between query and docs.
        # type: dict(qid, list( (docid, score) ))
        start = time.time()
        self.ranked_docs = self.rankDocs()
        print("{} seconds ranking docs...".format(time.time() - start))

        # write the ranked docs per query on disk
        # in specified format.
        self.dump()

    def tfidfVectorMod(apply_on):
        """
            reads from self.sparse_tfidf_document/queries
            depending on apply_on value being: doc/query
            returns modulus of tfidf vector
        """
        tfidf_mod = {}
        return tfidf_mod

    def evalInvDocFreq(self):
        """
            reads from self.data.posting_list
            returns idf for each term in vocab
                    type: dict(string : double)
            idf(term) = log(1 + number_of_docs/number_of_docs_containing_term)
            faster way: https://stackoverflow.com/questions/12282232/how-do-i-count-unique-values-inside-a-list
        """
        number_of_docs = len(self.data.docids)
        idf = {}
        for word in self.data.vocab:
            idf[word] = math.log(1 + number_of_docs/len(set(self.data.posting_list[word])))
        return idf

    def sparseTfIdfDocument(self):
        """
            reads from self.idf, self.posting_list
            returns sparse tfidf vector for all docs
                    type: dict(string : dict(string : double))
            tf(term, doc) = 1 + log(freq(term, doc))
        """
        tfidf = dict(zip(self.data.docids, [{} for _ in range(len(self.data.docids))]))
        # find freq of words in docs
        for word in self.data.posting_list:
            for docid in self.data.posting_list[word]:
                if tfidf[docid].get(word) == None:
                    tfidf[docid][word] = 1
                else:
                    tfidf[docid][word] += 1
        # find term freq = 1 + log(freq)
        # tfidf: mul tf and idf for each entry
        tfidf = {docid: {k: ((1.0 + math.log(v))*(self.idf[word]))\
                    for k, v in sub_dict.items()}\
                        for docid, sub_dict in tfidf.items()}
        # normalizing tfidf vector for all docs
        norm = {docid: math.sqrt(sum(v ** 2 for v in sub_dict.values())) for docid, sub_dict in tfidf.items()}
        tfidf = {docid: {word: tfidf_value/norm[docid]\
                    for word, tfidf_value in sub_dict.items()}\
                        for docid, sub_dict in tfidf.items()}
        return tfidf

    def sparseTfIdfQueries(self):
        """
            reads from self.data.tokenized_queries, self.idf
            returns sparse tfidf vector for all queries
                    type: dict(query_id, dict(string : double))
        """
        tfidf = {}
        return tfidf

    def rankDocs(self):
        """
            reads from  self.sparse_tfidf_document, 
                            type: dict(doc_id, dict(word, tfidf))
                        self.queries,
                            type: dict(query_id(int), list(words(=string))) 
                        self.cutoff
            returns ranked list of documents == cutoff
                    for all queries
                    type: dict(qid, list((docid, score)))
        """
        retrieved_docs = {}
        for qid, words in self.queries.items():
            # type(docs) = dict(docid, score)
            docs = {docid : sum(v for k, v in tfidf_dict.items() if k in words )\
                        for docid, tfidf_dict in self.sparse_tfidf_document.items()}
            # 
            retrieved_docs[qid] = [(docid, score) for docid, score in Counter(docs).most_common(self.cutoff)]
        return retrieved_docs

    def dump(self):
        """
            reads from self.ranked_docs
            writes to disk in a format accepted by trec_eval tool
            format: qid Q0 docid rank score FIRSTRUN
        """
        with open(self.result_file, 'w') as file:
            for qid, docs in self.ranked_docs.items():
                file.writelines([str(qid) + " Q0 " + docid + " 1 " + str(score) + " FIRSTRUN\n"\
                                    for docid, score in docs])
