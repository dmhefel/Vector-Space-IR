from pathlib import Path
import argparse, shelve
from flask import Flask, render_template, request
from utils import load_wapo
from inverted_index import build_inverted_index, query_inverted_index
from text_processing import TextProcessing
app = Flask(__name__)

text_processor = TextProcessing.from_nltk()
k = 50



data_dir = Path("pa4_data")
wapo_path = data_dir.joinpath("wapo_pa4.jl")
#wapo_path = data_dir.joinpath("test_corpus.jl")
shelve_index = str(data_dir.joinpath('hw4_shelf_full'))
doc_length_shelve = shelve_index+"_doc_len"
wapo_docs = {
    doc["id"]: doc for doc in load_wapo(wapo_path)
}  # comment out this line if you use the database
#global variables to keep track of the query term and results to see if new search should be performed or not
query = ""
result_docs = [[],[],[]]
query_idf = ""
db_terms = []
doc_lists_for_terms = []
# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST", "GET"])
def results():
    # get global variables
    global query
    global result_docs
    global query_idf
    global db_terms
    global doc_lists_for_terms
    # query; can be empty when showing results
    query_text = request.form["query"]  # Get the raw user query from home page
    # mpty results list handles if empty search term (this prevents breaking the app)
    results = []
    db = shelve.open(shelve_index)
    # if query is not empty (user entered new query)
    if query_text:
        # query_idf for printing in results page
        query_idf = ""
        db_terms = [] #get the normalized tokens to get doc lists
        doc_lists_for_terms = []
        for term in set(query_text.split()):
            if text_processor.normalize(term) in db:
                db_terms.append(text_processor.normalize(term))
                query_idf += term + ' ' + str(round(
                    text_processor.idf(len(db), len(db[text_processor.normalize(term)])), 4)) + " "
        #get doc lists
        for term in db_terms:
            docs = [x for (x,y) in db[term]]
            doc_lists_for_terms.append(docs)

        # set global variable for query as new query term
        query = query_text
        # do the search in shelf and save as global variable
        docs = query_inverted_index(query, k, shelve_index, doc_length_shelve)
        result_docs = docs
        d = docs[0]
        # split resulting doc list into lists of 8
        results = [d[i:i + 8] for i in range(0, len(d), 8)]
    # first page
    page = 1
    # starting number for listing
    s = ((page - 1) * 8 + 1)
    # next page num
    next = page + 1
    # if there is a next page
    hasnext = False
    if len(results) > page:
        hasnext = True
    # turn id nums from results list to the actual docs
    r = []
    if results:
        res = results[page - 1]
        for tup in res:
            d = wapo_docs[tup[1]]
            #add cosine similarity to dictionary
            d['cossim'] = round(tup[0], 4)
            #add the included terms to dictionary
            d['terms'] = ""
            for i in range(len(doc_lists_for_terms)):
                if tup[1] in doc_lists_for_terms[i]:
                    d['terms'] += query_idf.split()[i*2] if len(query_idf.split())>=i*2-1 else ''
                    d['terms'] += " "
            r.append(d)

    #join stopwords into one string if tehre are any
    stopwords = " ".join(result_docs[1])
    # join unk into one string if any
    unk = " ".join(result_docs[2])
    db.close()
    return render_template("results.html", query=query_text, idf=query_idf, result=r, page=page, next_page=next, start=s,
                           hasnext=hasnext, stop=stopwords, unknown=unk)  # add variables as you wish


# "next page" to show more results
@app.route("/results/<query_text>/<int:page_id>", methods=["POST", "GET"])
def next_page(page_id, query_text):
    # get global variables
    global query
    global result_docs
    global query_idf
    # get doc list
    d = result_docs[0]
    # split that list into 8s
    results = [d[i:i + 8] for i in range(0, len(d), 8)]
    # get page id
    page = page_id
    # starting number for listing
    s = ((page - 1) * 8 + 1)
    # next page num
    next = page + 1
    # sets variable for if there is next page
    hasnext = False
    if len(results) > page:
        hasnext = True
    # turn nums in doc list from id nums to actual docs
    r = []
    if results:
        res = results[page - 1]
        for tup in res:
            d = wapo_docs[tup[1]]
            # add cosine similarity to dictionary
            d['cossim'] = round(tup[0], 4)
            # add the included terms to dictionary
            d['terms'] = ""
            for i in range(len(doc_lists_for_terms)):
                if tup[1] in doc_lists_for_terms[i]:
                    d['terms'] += query_idf.split()[i * 2] if len(query_idf.split()) >= i * 2 - 1 else ''
                    d['terms'] += " "
            r.append(d)
    # join stopwords into one string
    stopwords = " ".join(result_docs[1])
    # join unk into one string
    unk = " ".join(result_docs[2])
    return render_template("results.html", query=query_text, idf = query_idf, result=r, page=page, next_page=next, start=s,
                           hasnext=hasnext, stop=stopwords, unknown=unk)  # add variables as you wish


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    # get doc and appropriate info to pass to html
    doc = wapo_docs[doc_id]
    txt = doc['content_str']
    title = doc['title']
    auth = doc["author"]
    date = doc['published_date']

    return render_template("doc.html", Text=txt, Author=auth, Title=title, Date=date)  # add variables as you wish


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean IR system")
    parser.add_argument("--build")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_inverted_index(
            wapo_path,
            str(data_dir.joinpath(args.build)),
            str(data_dir.joinpath(args.build)) + "_doc_len",
        )
    if args.run:
        app.run(debug=True, port=5000)
