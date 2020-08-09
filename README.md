Matrix Factorization with e-commerce data example
=================================================

Contents
--------

1.	[Introduction](#introduction)
2.	[Dataset](#dataset)
3.	[Results](#results)
4.	[Todos](#todos)

Introduction
------------

---

A simple example to illustrate how the Matrix Factorization model can be used to build a Recommender System.

-	Matrix_Factorization.ipynb: preprocessing data, training two kinds of Matrix Factorization models(SVD & NMF), and evaluating trained model
-	recsystools.py: some tools that are useful to build and evaluate Recommender System models

In this example, we are going to predict which product a customer might be interested in based on past purchase records from an e-commerce site. Different from the usual movielens example, we are trying to predict user preference using implicit feedback data.

for a simple and intuitive explanation about Matrix Factorization, I recommend this amazing video of Luis Serrano:

https://www.youtube.com/watch?v=ZspR5PZemcs&t=5s

![PCA on Digits Dataset](https://www.researchgate.net/profile/Jun_Xu27/publication/321344494/figure/fig1/AS:702109309751298@1544407312766/Diagram-of-matrix-factorization.png)

Dataset
-------

---

Download dataset via URL below:

```bash
https://www.kaggle.com/carrie1/ecommerce-data
```

This dataset contains 540k transaction records of 4372 customers and 4070 products.

Results
-------

---

parameter k refers to the dimension of the latent feature space. values are rounded to 2 decimal places.

| methods     | NDCG@10   | Recall@10 | training time |
|-------------|:---------:|:---------:|:-------------:|
| SVD (k=200) | **0.120** | **0.104** |     6.5s      |
| NMF (k=50)  |   0.081   |   0.071   |     16.8s     |

Todos
-----

---

I am planning to make more examples treating Recsys models using VAE(Variational AutoEncoder), FM(Factorization Machine) algorithms.

### References

https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
