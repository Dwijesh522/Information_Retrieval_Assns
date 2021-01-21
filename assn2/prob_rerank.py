import sys
import nltk
import gzip
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

# global variable
stop_words = set(stopwords.words('english'))

def error():
    print("Error in command line arguments. Refer Assignment for synopsis")

def readQueryWords(qFile_path, direct):
    """direct is pass by ref output dict: qid -> [word]"""
    with gzip.open(qFile_path, 'rt') as qfile:
        while(True):
            line = qfile.readline()
            if not line:    break
            qwords = line.split()
            qid = qwords[0]
            qwords = [w for w in qwords[1:] if not w in stop_words]
            direct[qid] = qwords

def readReleventDocs(top100File_path, relevent_docs):
    """relevent_docs is pass by ref output dict: docid -> [qids]"""
    with gzip.open(top100File_path, 'rt') as rdoc_file:
        while(True):
            line = rdoc_file.readline()
            if not line:    break
            words = line.split()
            qids = relevent_docs.get(words[2])
            if qids == None:
                relevent_docs[words[2]] = [words[0]]
            else:
                relevent_docs[words[2]] += [words[0]]

def readDocs(relevent_docs, cFile_path, indirect, doc_lengths):
    """ indirect is pass by ref output dict: qid -> words,
        doc_lengths is pass by ref output dict: docid -> length
        invarient: each line in cFile_path is a new doc"""
    with gzip.open(cFile_path, 'rt') as docs:
        while(True):
            line = docs.readline()
            if not line:    break
            docid, web, body = line.split('\t', maxsplit=2)
            doc_lengths[docid] = len(body)
            if docid in relevent_docs:
                words = [w for w in body.split() if not w in stop_words]
                for qid in relevent_docs[docid]:
                    indirect[qid] += words

def updateQ(d, Q):
    """ type(d) = dict(string, list(string))
        type(Q) = set(string)"""
    for sub_list in d.values():
        Q.update(sub_list)

def updateTerm(cFile_path, Term):
    """Term is pass by ref output nested dict: (word->(docid->freq))"""
    with gzip.open(cFile_path, 'rt') as collection:
        while(True):
            line = collection.readline()
            if not line:    break
            docid, web, body = line.split('\t', maxsplit=2)
            for word in body.split():
                if not word in stop_words:
                    if word in Term:
                        if docid in Term[word]:
                            Term[word][docid] += 1
                        else:
                            Term[word][docid] = 1
                    else:
                        Term[word] = {docid:1}

def getRelevenceWeight(expansion_lim, N, direct, indirect, Q, invidx, relevent_docs):
    """ type(direct) = dict(qid, [words])
        type(indirect) = dict(qid, [words])
        type(Q) = set(words)
        type(invidx) = dict(word, dict(docid, freq))
        type(relevent_docs) = dict(docid, [qid])
        
        return sparse matrix of RW for words in direct + newly expanded words"""

    # unique words from direct
    unique_dWords = set()
    updateQ(direct, unique_dWords)
    direct_indexed_words = dict(zip(unique_dWords, range(len(unique_dWords))))
    #unique words from indirect
    unique_idWords = set()
    updateQ(indirect, unique_idWords)
    indirect_indexed_words = dict(zip(unique_idWords, range(len(unique_idWords))))
    _indirect_indexed_words = dict(zip(range(len(unique_idWords)), unique_idWords))
    
    # indexing the query
    indexed_qids = dict(zip(direct.keys(), range(len(direct))))

    r_direct = lil_matrix((len(unique_dWords), len(direct)), dtype=np.float32)
    n_direct = lil_matrix((len(direct), len(unique_dWords)), dtype=np.float32)
    
    r_indirect = lil_matrix((len(unique_idWords), len(indirect)), dtype=np.float32)
    n_indirect = lil_matrix((len(indirect), len(unique_idWords)), dtype=np.float32)

    # n_direct
    for qid in direct:
        for word in set(direct[qid]):
            if word in invidx:
                n_direct[indexed_qids[qid], direct_indexed_words[word]] +=\
                        len(invidx[word])
    # r_direct
    for word in unique_dWords:
        if word in invidx:
            for docid in invidx[word]:
                if docid in relevent_docs:
                    for qid in relevent_docs[docid]:
                        r_direct[direct_indexed_words[word], indexed_qids[qid]] += 1
 
    # n_indirect
    for qid in indirect:
        for word in set(indirect[qid]):
            if word in invidx:
                n_indirect[indexed_qids[qid], indirect_indexed_words[word]] +=\
                        len(invidx[word])
    # r_indirect
    for word in unique_idWords:
        if word in invidx:
            for docid in invidx[word]:
                if docid in relevent_docs:
                    for qid in relevent_docs[docid]:
                        r_indirect[indirect_indexed_words[word], indexed_qids[qid]] += 1
    
    # preparing the parameters
    r_direct = r_direct.todense().T
    n_direct = n_direct.todense()
    r_indirect = r_indirect.todense().T
    n_indirect = n_indirect.todense()
    R = 100
    
    # creating mask: REMOVE IF NOT NEEDED
    mask_r_direct = r_direct!=0
    mask_n_direct = n_direct!=0
    mask_r_indirect = r_indirect!=0
    mask_n_indirect = n_indirect!=0

    p0 = 0.5 + 10 ############################### remove this for original dataset
    # qid X unique words in direct
    rw_direct = np.log(\
                        np.multiply(r_direct+p0, N-n_direct-R+r_direct+p0)/\
                        np.multiply(n_direct-r_direct+p0, R-r_direct+p0)\
                )
    # qid x unique words in indirect
    rw_indirect = np.log(\
                        np.multiply(r_indirect+p0, N-n_indirect-R+r_indirect+p0)/\
                        np.multiply(n_indirect-r_indirect+p0, R-r_indirect+p0)\
                )
    
    # finding top <expansion_lim> words per query
    # indices.shape: qid x expansion_lim
    indices = np.argpartition(rw_indirect, -expansion_lim, axis=1)[:, -expansion_lim:]
    r_indices, c_indices = indices.shape

    # need consistency: rw_direct*, direct_indexed_words*
    # shape(rw_direct*) = qid x (unique(old+new) words)
    # shape(direct_indexed_words*) = dict(unique(old+new)word, int)
    temp_indices = []
    for i in range(r_indices):
        for j in range(c_indices):
            temp_ind = indices[i, j]
            temp_word = _indirect_indexed_words[temp_ind]
            if not temp_word in direct_indexed_words:
                direct_indexed_words[temp_word] = len(direct_indexed_words)
                # not an inplace operation
                temp_indices += [temp_ind]
                
    rw_direct = np.hstack((rw_direct, rw_indirect[:, temp_indices]))
    return rw_direct, direct_indexed_words, indexed_qids

def getPartialScore(invidx, indexed_terms, DL, PartialScore):
    """
        type(invidx) = dict(word, dict(docid, freq))
        type(indexed_terms) = dict(word, index)
        type(DL) = dict(docid, length)
        shape(PartialScore) = words x docid
    """
    dlavg = sum(list(DL.values()))/len(DL)
    k1 = 1.5
    b = 0.75

    # fixing the indices of docids
    indexed_docids = dict(zip(DL.keys(), range(len(DL))))
    for word in indexed_terms:
        if word in invidx:
            for docid, freq in invidx[word].items():
                PartialScore[indexed_terms[word], indexed_docids[docid]] = \
                        (freq * (k1+1))/\
                            ( k1*((1-b) + b*(DL[docid]/dlavg)) + freq)

def dump(expansion_lim, Score, indexed_qids, _indexed_docids):
    """
        format = qid Q0 docid rank score STRING
        shape(Score) = qid x docid
        type(indexed_qids) = dict(qid, index)
        type(_indexed_docids) = dict(index, docid)
    """
    with open('reranked_prob_{}.txt'.format(expansion_lim), 'w') as output:
        indices = np.argpartition(Score, -100, axis=1)[:, -100:].tolist()
        for qid in indexed_qids:
            qid_index = indexed_qids[qid]
            for docid_index in indices[qid_index]:
                docid = _indexed_docids[docid_index]
                output.write("{} Q0 {} 1 {} STRING\n".format(qid, docid, Score[qid_index, docid_index]))
        

if __name__ == '__main__':
    # command line error
    if (len(sys.argv) != 5):    error()
    # fetching command line args
    _, qFile_path, top100File_path, cFile_path, expansion_lim = sys.argv
    expansion_lim = int(expansion_lim)

    direct = {} # qid -> [words]                        ##### words are not unique
    readQueryWords(qFile_path, direct)
    relevent_docs = {} # docid -> qid
    readReleventDocs(top100File_path, relevent_docs)
    indirect = dict(zip(direct.keys(), ([] for _ in range(len(direct))))) # qid -> [words]  #####
    DL = {} # type(doc_lengths) = dict(docid, length)
    readDocs(relevent_docs, cFile_path, indirect, DL)

    Q = set() # unique direct and indirect words
    updateQ(direct, Q)
    updateQ(indirect, Q)

    invidx = {} # (word -> (docid -> freq))
    updateTerm(cFile_path, invidx)

    # shape(RW) = qid x (words in terms)
    # type(terms) = dict(words, index)
    # type(indexed_qids) = dict(qid, index)
    RW, indexed_terms, indexed_qids= \
        getRelevenceWeight(expansion_lim, len(DL), direct, indirect, Q, invidx, relevent_docs)
    print("RESOLVE: value changes every time: ", RW.shape)

    #%%%%%%%%% delete unnecessary data all over the program  %%%%%%%%%%%%%%%%
    # shape(PartialScore) = words x docids
    PartialScore = np.matrix(np.zeros((len(indexed_terms), len(DL)), dtype=np.float32))
    getPartialScore(invidx, indexed_terms, DL, PartialScore)

    # shape(Score) = qid * docid
    Score = RW.dot(PartialScore)
    print("RESOLVE: values of Score seems to be same in column")

    _indexed_docids = dict(zip(range(len(DL)), DL.keys()))
    dump(expansion_lim, Score, indexed_qids, _indexed_docids)
