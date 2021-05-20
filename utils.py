from typing import Dict, Union, Iterator
from pathlib import Path
import functools
import os
import time
import re
import json
from datetime import datetime as dt

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_t = time.perf_counter()
        f_value = func(*args, **kwargs)
        elapsed_t = time.perf_counter() - start_t
        mins = elapsed_t // 60
        print(
            f"'{func.__name__}' elapsed time: {mins} minutes, {elapsed_t - mins * 60:0.2f} seconds"
        )
        return f_value

    return wrapper_timer


def load_wapo(wapo_jl_path: Union[str, os.PathLike]) -> Iterator[Dict]:
    """
    It should be similar to the load_wapo in HW3 with two changes:
    - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
    - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datatime package to do this.
    """
    # open the jsonline file
    with open(wapo_jl_path) as f:
        # get line
        line = f.readline()
        # id number
        id = 0
        # while line not empty
        while line:
            # load line as json
            article = json.loads(line)
            # process content by getting each sanitized html sentence and adding to list.
            content = []
            for sentence in article['contents']:
                if sentence is not None and 'content' in sentence and sentence['type'] == 'sanitized_html':
                    content.append(sentence['content'])
            # turn list of sentences into one string
            content = ' '.join(content)
            # use regex to get rid of html elements
            content = re.sub('<[^<]+?>', '', content)
            #process date
            dateinfo = dt.fromtimestamp(article['published_date'] / 1000)
            date = str(dateinfo.year) + "/" + str(dateinfo.month) + "/" + str(dateinfo.day)
            # create dict for the article
            articledict = {'id': id,
                           'title': article["title"] if article['title'] is not None else '',
                           "author": article["author"] if article['author'] is not None else '',
                           "published_date": date,
                           "content_str": content if content is not None else ''}
            # increment id #
            id += 1
            # get next line
            line = f.readline()
            # yield the dictionary (generator object)
            yield articledict


if __name__ == "__main__":
    pass


# data_dir = Path("pa4_data")
# wapo_path = data_dir.joinpath("test_corpus.jl")
# wapo_docs = {doc["id"]: doc for doc in load_wapo(wapo_path)}
# wapo = load_wapo(wapo_path)
# print(next(wapo, ''))
# print(type(wapo))
#
# print("\n\n\n")