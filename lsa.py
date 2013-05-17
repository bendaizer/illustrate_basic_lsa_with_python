# -*- coding: utf-8 -*-

import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def multiply(*args):
    """takes custom number of numpy arrays and multiplies them

    Arguments:
    take as many numpy arrays as necessary
    """
    i = 0
    res = 1
    while i < len(args):
        M = args[i]
        i += 1
        res = np.dot(res, M)
    return res


def cosine(x, y):
    """returns cosine of angle between x et y"""
    return np.dot(x, y)/sqrt(np.dot(x, x))/sqrt(np.dot(y, y))


def build_terms(docs, stopwords):
    """build dictionary of words from docs, ignoring list of stopwords """
    terms = list(set([item.lower() for s in docs for item in s.split(" ")
                      if item.lower() not in stopwords]))
    terms.sort()
    terms = dict((key, value) for (value, key) in enumerate(terms))
    return terms


def build_M(terms, docs):
    """take a list of string docs, and a dict for terms
       extracts vector of words and build term-doc matrix"""
    docs_split = [doc_list.lower().split(" ") for doc_list in docs]
    n_docs = len(docs)
    n_terms = len(terms)

    M = np.zeros((n_terms, n_docs))

    for i, doc in enumerate(docs_split):
        for term in doc:
            if term in terms:
                M[terms.get(term), i] += 1
    return M


def tfidf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead

    Arguments:
    M : numpy 2d float array
    """

    return tf(M)*idf(M)


def idf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead

    Arguments:
    M : numpy 2d float array
    """
    n_terms = M.shape[0]
    n_docs = float(M.shape[1])
    Mtfidf = np.zeros((n_terms, n_docs))

    for term in range(n_terms):
        dt = float(np.count_nonzero(M[term]))
        Mtfidf[term] = log(n_docs/dt) if dt != 0 else 0
    return Mtfidf


def tf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead

    Arguments:
    M : numpy 2d float array
    """

    n_terms = M.shape[0]
    n_docs = M.shape[1]
    Mtf = np.zeros((n_terms, n_docs))

    for doc in range(n_docs):
        # Mtf[:, doc] = M[:, doc]/M[:, doc].sum()
        Mtf[:, doc] = M[:, doc]/M[:, doc].max()
    return Mtf


def scatter(U, V, labels):
    plt.scatter(U, V)
    for label, x, y in zip(labels, U, V):
        plt.annotate(
            label,
            xy=(x, y), xytext=(0.9, 9),
            textcoords="offset points", ha="right", va="bottom")




# def main():

# densier numpy array printing
np.set_printoptions(precision=3)

# documents = [
#     "Human machine interface for lab abc computer applications",
#     "A survey of user opinion of computer system response time",
#     "The EPS user interface management system",
#     "System and human system engineering testing of EPS",
#     "Relation of user perceived response time to error measurement",
#     "The generation of random binary unordered trees",
#     "The intersection graph of paths in trees",
#     "Graph minors IV Widths of trees and well quasi ordering",
#     "Graph minors A survey"
# ]

# docs_label = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]

stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])

#==========
documents = [
    "chat souris",
    "chat souris animaux",
    "souris clavier",
    "clavier ordinateur"
]
docs_label = ["A1", "A2", "B1", "B2"]  # chat souris etc
<<<<<<< HEAD

terms_label = build_terms(documents, stopwords)

=======

terms_label = build_terms(documents, stopwords)

>>>>>>> ce896f7cf4d9fb29355ce68e174b897d8b010a80
# ==========
# using custom terms
# terms_label = ["human",
#          "interface",
#          "computer",
#          "user",
#          "system",
#          "response",
#          "time",
#          "EPS",
#          "survey",
#          "trees",
#          "graph",
#          "minors"]


#==========
# documents = [
#     "A A A",
#     "B B C A C",
#     "A B A C",
#     "A A A B",
#     "X X Y",
#     "Y Y X Z",
#     "X X Z Z"
# ]

# terms_label = build_terms(documents, stopwords)


terms = dict((key.lower(), value) for (value, key) in enumerate(terms_label))


M = build_M(terms, documents)
# M = tfidf(M)

U, s, V = np.linalg.svd(M)
S = np.zeros(M.shape)

# # S[:s.size, :s.size] = np.diag(s)
S[:s.size, :s.size] = np.diag([k if i < 3 else 0 for (i, k) in enumerate(s)])


# scatter(U[:, 0], U[:, 1], terms_label)
# scatter(V[:, 1], V[:, 2], docs_label)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(U[:, 0], U[:, 1], U[:, 2])

plt.show()

# # if np.allclose(M, np.dot(U, np.dot(S, V))):
# #     print "SVD OK"

# # print ""
# # print "topic 1"
# # print [(i, j) for (i, j) in zip(np.dot(U, S)[:, 0], terms) if abs(i)>0.01]  # if abs(i) > 0.1]
# # print ""
# # print "topic 2"
# # print [(i, j) for (i, j) in zip(np.dot(U, S)[:, 1], terms) if abs(i)>0.01]  #if abs(i) > 0.1]


# =======
# if __name__ == '__main__':
#     main()



A = np.array([[0,1,0,0,1],[0,1,0,0,1],[3,2,2,0,1],[0,0,1,1,0],[0,0,1,1,1],[0,0,0,0,0]])
A = A.astype(float)
