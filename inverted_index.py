from typing import Union, List, Tuple, Dict
import os
import shelve
import heapq
from utils import timer, load_wapo
from text_processing import TextProcessing
import math

text_processor = TextProcessing.from_nltk()


def get_doc_vec_norm(term_tfs: List[float]) -> float:
    """
    helper function, should be called in build_inverted_index
    compute the length of a document vector
    :param term_tfs: a list of term weights (log tf) for one document
    :return:
    """
    return math.sqrt(sum([i**2 for i in term_tfs]))


@timer
def build_inverted_index(
    wapo_jl_path: Union[str, os.PathLike],
    index_shelve_path: str,
    doc_vec_norm_shelve_path: str,
) -> None:
    """
    load wapo_pa4.jl to build two shelve files in the provided path

    :param wapo_jl_path:
    :param index_shelve_path: for each normalized term as a key, the value should be a list of tuples;
        each tuple stores the doc id this term appear in and the term weight (log tf)
    :param doc_vec_norm_shelve_path: for each doc id as a key, the value should be the "length" of that document vector
    :return:
    """
    #open doc_vec_norm_shelve
    doc_vec_shelf = shelve.open(doc_vec_norm_shelve_path)
    # create a list of tuples of form (token, docID, term weight)
    tuples = []
    # call generator object of wapo
    wapo = load_wapo(wapo_jl_path)
    article = next(wapo, '')
    while article:
        # get set of tokens from the doc
        tokens = text_processor.get_normalized_tokens(article['title'], article['content_str'])
        #create list of tfs
        tfs = []
        #process tokens
        tf = 1
        if tokens:
            term = tokens[0]
            for i in range(1,len(tokens)):
                if tokens[i] == term:
                    tf+=1
                else:
                    tf = text_processor.tf(tf)
                    tfs.append(tf)
                    tuples.append((term, article['id'], tf))
                    term = tokens[i]
                    tf = 1
            tf = text_processor.tf(tf)
            tfs.append(tf)
            tuples.append((term, article['id'], tf))

            #get doc_vec_norm and add to shelf
            doc_vec_shelf[str(article['id'])] = get_doc_vec_norm(tfs)
        # get next doc
        #print(article['id'])
        article = next(wapo, '')
    # sort tuple list for adding to shelf
    tuples.sort()
    # open shelf
    d = shelve.open(index_shelve_path)
    # get term
    term = tuples[0][0]
    # get the list of docs that term is in
    docs = [(tuples[0][1], tuples[0][2])]
    # iterate through tuple list
    for i in range(1, len(tuples)):
        # if tuple has the same term, just append docID to list of docs
        if tuples[i][0] == term:
            docs.append((tuples[i][1],tuples[i][2]))
        # if diff term, add term to shelf and reset doc list with new term.
        else:
            d[term] = docs
            docs = [(tuples[i][1], tuples[i][2])]
            term = tuples[i][0]
    d[term] = docs
    d.close()


def parse_query(
    query: str, shelve_index: shelve.Shelf
) -> Tuple[List[str], List[str], List[str]]:
    """
    helper function, should be called in query_inverted_index
    given each query, return a list of normalized terms, a list of stop words and a list of unknown words separately

    :param query:
    :param shelve_index:
    :return:
    """
    shelf = shelve.open(shelve_index)
    norm_terms = []
    stop = []
    unk = []
    for term in query.split():
        t = text_processor.normalize(term)
        if t:
            if t in shelf:
                norm_terms.append(t)
            else:
                unk.append(term)
        else:
            stop.append(term)
    return (norm_terms,stop,unk)


def top_k_docs(doc_scores: Dict[int, float], k: int) -> List[Tuple[float, int]]:
    """
    helper function, should be called in query_inverted_index
    given the doc_scores, return top k doc ids and corresponding scores using a heap
    :param doc_scores: a dictionary where doc id is the key and cosine similarity score is the value
    :param k:
    :return: a list of tuples, each tuple contains (score, doc_id)
    """

    #create the heap
    heap = []
    #for items in dict, push if still room, else push then pop to maintain k items.
    for key,val in doc_scores.items():
        if len(heap)<k:
            heapq.heappush(heap, (val,key))
        else:
            heapq.heappushpop(heap, (val,key))
    return heap

def query_inverted_index(
    query: str, k: int, shelve_index: shelve.Shelf, doc_length_shelve: shelve.Shelf
) -> Tuple[List[Tuple[float, int]], List[str], List[str]]:
    """
    disjunctive query over the shelve_index
    return a list of matched documents (output from the function top_k_docs), a list of stop words and a list of unknown words separately
    :param query:
    :param k:
    :param shelve_index:
    :param doc_length_shelve:
    :return:
    """
    #process query
    processed_query = parse_query(query, shelve_index)
    terms = processed_query[0]
    #create the vector for the query... starts with just tf, gets updated with idf later
    if len(terms)==len(set(terms)): #most common... each word in query is used once
        w_tq = [text_processor.tf(1)]*len(terms) #vector of 1s
    else: #if someone did a query with repeated words, such as best car foreign car
        w_tq = []
        new_terms = list(set(terms))
        for term in new_terms:
            w_tq.append(text_processor.tf(terms.count(term)))
        terms = new_terms
    #initialize doc_scores
    doc_scores = {}

    #open shelf
    db = shelve.open(shelve_index)
    N = len(db)
    #go through term list and add doc lists to doc_scores
    for i in range(len(terms)):
        doc_list = db[terms[i]]
        idf = text_processor.idf(N, len(doc_list)) #updates w_tq of query to be tf-idf
        w_tq[i] *= idf
        for tup in doc_list: #if doc in dict already, add q x d for term, else add to dict
            if tup[0] in doc_scores:
                doc_scores[tup[0]] += tup[1]*idf*w_tq[i]
            else:
                doc_scores[tup[0]] = tup[1]*idf*w_tq[i]

    #normalize length by using doc_vec shelf
    doc_vec_shelf = shelve.open(doc_length_shelve)
    for key,val in doc_scores.items():
        doc_scores[key] /= doc_vec_shelf[str(key)]
    doc_vec_shelf.close()
    #get top k from doc score dict
    top_k = top_k_docs(doc_scores, k)
    top_k.sort(reverse=True)
    #close shelf
    db.close()
    return (top_k, processed_query[1], processed_query[2])


if __name__ == "__main__":
    pass
