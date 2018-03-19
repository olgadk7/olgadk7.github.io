---
title: "Reading List: Collaborative Modelling for Recommending Scientific Aricles, Chong Wang and David M. Blei, 2001"
layout: post
tag: papers
blog: true
author: Olga Krieger
summary: "Reading List: recaps of important data science papers."
permalink: blog/reading-list-2011wangblei
---

### *This is the second post in Reading List series where I recap important papers as part of continuing my data science education.*

I wanted to study this paper because it describes the recommendation engine used by the New York Times. The domain of this paper is online archives of scientific articles which researchers use to find relevant papers, but my recap can be generalised to other recommendation contexts by substituting researchers with users and articles with items.

Within the domain-specific context of the paper, the disadvantages of the traditional way of finding new articles via citations in other articles are:
- it limits researchers to heavily cited papers, 
- it limits researchers to papers within one field and
- researchers may miss a relevant paper because it was also missed by the authors of the papers they read.

The disadvantages of a more general *keyword search*, are:
- forming queries can be difficult as a researcher may not know what to look for; 
- search is mainly based on content, while good articles are also those that many others found valuable; and 
- search is only good for directed exploration, while many researchers would also like a “feed” of new and interesting articles.

Recently however online sharing communities have been recommending new material via *either* 
1. *collaborative filtering* (the traditional approach to recommendation where items are recommended to a user based on other users with similar patterns of selected items)
2. probabilistic *topic modeling* (also referred to as content filtering; these are ‘algorithms that are used to discover a set of “topics” from a large collection of documents, where *a topic is* a distribution over terms that is biased around those associated under a single theme. Topic models provide an interpretable low-dimensional representation of the documents. They have been used for tasks like corpus exploration, document classification, and information retrieval.)

The two types of recommendation problems associated with those are:
1. classical *matrix factorization* solution to recommendation (a latent factor method that performs well, but there are two disadvantages: first, the learnt latent space is not easy to interpret; second, matrix factorization only uses information from other users—it cannot generalize to completely unrated items.)
2. latent Dirichlet allocation (*LDA*, the simplest topic model of text).

The two types of recommendation prediction: 
- *in-matrix* (making recommendations about those articles that have been rated by at least one user in the system) 
- *out-of-matrix* (making recommendations about those articles that have never been rated). Traditional collaborative filtering algorithms cannot make predictions about these articles because those algorithms only use information about other users’ ratings. A recommender system that cannot handle out-of-matrix prediction cannot recommend newly published papers to its users.

The authors propose an algorithm - collaborative topic regression (CTR) - that combines the merits of the two traditional methods. It uses the same type of data, the other users’ libraries and the content of the articles.

As a judging criteria, a good recommendation engine for them is one that values
- older foundational/classic works
- newer undiscovered/state-of-the-art works
- exploratory variables, ie a summary and description of each user’s preference profile based on the content of the articles that he or she likes - something like that allows to ‘connect similar users to enhance the community, and indicate why we are connecting them. Further, we can describe articles in terms of what kinds of users like them.’

Wang and Blei start by fitting a model that uses the latent topic space to explain both the observed ratings and the observed words. This model though cannot distinguish topics for explaining recommendations from topics important for explaining content. CTR however can detect this difference. The key property in CTR lies in how the item latent vector is generated: authors assume it’s close to topic proportions, but could diverge from it if it has to. This expectation is a linear function - this is why the model is called collaborative topic regression.

As a result, CTR can 
1. find older papers that are important to other similar users 
2. find newly written papers whose content reflects the user’s specific interests
3. give interpretable representations of users and articles.

This approach works well relative to traditional matrix factorization methods and makes good predictions on completely unrated articles. Further, our algorithm provides interpretable user profiles that can be useful in real-world recommender systems.



**_Keywords:_** Recommendation engine, Collaborative Filter, Content Filter, Topic Model, Keyword Search, Latent Dirichlet Allocation
