import sys
import DocumentProcessing as dp

def error():
    print("Follow the following commandline format for running the program.")
    print("$ python3 invidx_cons.py collection_path indexfile")

if __name__ == '__main__':
    if (len(sys.argv) != 3):
        error()
        exit(0)

    directory_name = sys.argv[1]
    indexfile = sys.argv[2]

    document_processing_obj = dp.DocumentProcessing(directory_name, indexfile)
