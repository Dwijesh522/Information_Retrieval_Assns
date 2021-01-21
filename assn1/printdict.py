import json

posting_list = json.load(open('indexfile.idx'))

for word, posting in posting_list.items():
    print(word + ":" + str(len(set(posting))) + ":" + "-")
