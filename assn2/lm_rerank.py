import sys
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import time
import gzip
from functools import reduce
from operator import mul

"""list of stop words, and stemmer"""
STOP_WORDS = set(stopwords.words('english'))
PS = PorterStemmer()
MYU = 1

def error():
    """command line error"""
    print("\nexpected command line: ")
    print("python3 lm_rerank.py path_to/msmarco-docdev-queries.tsv path_to/msmarco-docdev-top100 path_to/docs.tsv uni/bi\n")

def preprocess(tokens):
    """ type(tokens) = list(words)
        apply lower case, stop word removal, stemming
        return list(words)
    """
    processed = []
    for word in tokens:
        lword = word.lower()
        if not lword in STOP_WORDS:
            processed += [PS.stem(lword)]
    return processed

def fetchTop100Docids(qid, top100_path):
    """ given qid, store top100 docid for that qid in list
        and return. So return type: list[docids]"""
    print("fetching top100 docids")
    top100_docids = []
    with gzip.open(top100_path, 'rt') as top100_file:
        while (len(top100_docids) < 100) :
            line = top100_file.readline()
            if not line : break
            _qid, _, docid, _, _, _ = line.split()
            if _qid == qid :
                top100_docids += [docid]
    return top100_docids

def getPostingListAndDL(top100_docids, coll_path):
    """ type(top100_docids) = list[docids]
        type(coll_path) = string
        type(p_list) = dict(word, dict(docid, freq))
        type(dl): dict(docid, doc length)"""
    print("calculating the posting list...")
    dl = dict(zip(top100_docids, (0 for _ in range(len(top100_docids))))) # doc length
    p_list = {} # posting list
    with gzip.open(coll_path, 'rt') as coll_file:
        while (True) :
            line = coll_file.readline()
            if not line :   break
            tokens = line.split()
            docid = tokens[0]
            if docid in top100_docids:
                """current doc belongs to top100 docs"""
                body_tokens = tokens[2:]
                processed_tokens = preprocess(body_tokens) # processing words
                dl[docid] = len(processed_tokens) # length of current doc
                for word in processed_tokens:
                    if word in p_list:
                        if docid in p_list[word]:
                            p_list[word][docid] += 1
                        else:
                            p_list[word][docid] = 1
                    else:
                        p_list[word] = {docid: 1}
    return p_list, dl

def P_w_given_M(N, word, docid, p_list, dl):
    "return p(w|M)"
    freq_w_d = 0
    n = 0
    if word in p_list:
        n = len(p_list[word])
        if docid in p_list[word]:
            freq_w_d = p_list[word][docid]
    return (freq_w_d + (MYU*(n+1/N)))/(dl[docid] + MYU)

def P_w(word, top100_docids, p_list, dl):
    "return p(w)"
    pw = 0
    pm = 1/len(top100_docids)
    pw += sum([P_w_given_M(len(top100_docids), word, docid, p_list, dl) for docid in top100_docids])
    pw *= pm
    return pw

def P_M_given_w(word, docid, top100_docids, p_list, dl):
    "retrn p(M|w)"
    N = len(top100_docids)
    pm = 1/N
    return (P_w_given_M(N, word, docid, p_list, dl) * P_w(word, top100_docids, p_list, dl))/pm

def getKLDivergence(total_docs, did, q_terms, p_list, dl, top100_docids):
    """ type(total_docs) = int
        type(did) = string, current docid
        type(p_list) = dict(word, dict(docid, freq))
        type(dl) = dict(docid, length)
        type(top100_docids) = list(docids)
        return E(P(w|R)*log(P(w|D))) """
    print("started to calculate kl divergence...")
    score = 0.0
    N = len(top100_docids)
    for word in p_list:
        jp_w_q = P_w(word, top100_docids, p_list, dl)
        for qi in q_terms:
            p_qi_given_w = 0
            for docid in top100_docids:
                p_qi_given_w += (P_M_given_w(word, docid, top100_docids, p_list, dl)*\
                                 P_w_given_M(N, qi, docid, p_list, dl))
            jp_w_q *= p_qi_given_w
        score += (jp_w_q * math.log(P_w_given_M(N, word, did, p_list, dl)))
    return score

def dump(qid, ranked, reranked_file):
    """ format: qid Q0 docid rank score STRING"""
    rank = 1
    for (docid, score) in ranked:
        reranked_file.write("{} Q0 {} {} {} RERANKED\n".format(qid, docid, str(rank), str(score)))
        rank += 1

def rerank(qid, q_terms, top100_path, coll_path, reranked_file):
    print("entered into reranking...")
    top100_docids = fetchTop100Docids(qid, top100_path)
    p_list, dl = getPostingListAndDL(top100_docids, coll_path)
    doc_scores = dict(zip(top100_docids, (0.0 for _ in range(len(top100_docids))))) # type = dict(docid, score)
    for did in top100_docids:
        score = getKLDivergence(len(top100_docids), did, q_terms, p_list, dl, top100_docids)
        doc_scores[did] = score
    """sort and dump"""
    ranked = sorted(doc_scores.items(), key=lambda kv:(kv[1], kv[0]))
    dump(qid, ranked, reranked_file)

if __name__ ==  '__main__':
    if len(sys.argv) != 5:  error(); exit(0)

    _, query_path, top100_path, coll_path, model = sys.argv
    count = 0
    reranked_file = open('reranked_lm.txt', 'w')
    with gzip.open(query_path, 'rt') as query_file:
        while (True) :
            count += 1
            """fetching qid, terms"""
            line = query_file.readline()
            if not line: break
            tokens = line.split()
            qid = tokens[0]
            terms = preprocess(tokens[1:])
            rerank(qid, terms, top100_path, coll_path, reranked_file)
    reranked_file.close()
