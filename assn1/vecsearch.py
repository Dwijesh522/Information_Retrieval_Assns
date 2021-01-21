import TfIdf
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-q", "--query", action="store", dest="query_path")
    parser.add_option("-k", "--cutoff", action="store", type='int', dest = "cutoff")
    parser.add_option("-o", "--output", action="store", dest = "resultfile")
    parser.add_option("-i", "--index", action = "store", dest = "posting_list_path")
    parser.add_option("-d", "--dict", action = "store", dest = "vocab_path")

    options, _ = parser.parse_args()
    options = vars(options)

    tfidf_obj = TfIdf.TfIdf(options['vocab_path'], options['posting_list_path'], options['query_path'],\
                            options['cutoff'], options['resultfile'])
