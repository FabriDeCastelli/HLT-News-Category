# HLT-News-Category
News classification task for Human Language Technologies course.

# Authors:
- [Fabrizio De Castelli](https://github.com/FabriDeCastelli)&nbsp;&nbsp;&nbsp;- M.Sc. in Artificial Intelligence, University of Pisa
- [Francesco Aliprandi](https://github.com/francealip)&nbsp;&nbsp;&nbsp;- M.Sc. in Artificial Intelligence, University of Pisa
- [Francesco Simonetti](https://github.com/francescoS01)&nbsp;&nbsp;- M.Sc. in Artificial Intelligence, University of Pisa
- [Marco Minniti](https://github.com/Marco-Minniti)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - M.Sc. in Artificial Intelligence, University of Pisa
- [Tommaso Di Riccio](https://github.com/tommasoDR)&nbsp;&nbsp;&nbsp;&nbsp;- M.Sc. in Artificial Intelligence, University of Pisa

# Abstract

This project concerns the multinomial classification of HuffPost news articles from 
[Kaggle dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset). We compare the performance of various models. As baseline Naive Bayes, 
Logistic Regression were considered, as intermediate bidirectional LSTM and as state-of-the-art
models the ones from BERT family and LLAMA3 LLM. 

# Guidelines for running the code

Recommended version of python is python==3.1.18.

To install the required libraries, run the following command:

```bash pip install -r requirements.txt```

For test execution, each model is implemented in a separate notebook file.
Notebooks are located in **src/test** folder. Each notebook is named with the model name.

For running the bidirLSTM experiment pre-trained embeddings are required. 
For this analysis several pre-trained embeddings were considered, but only those of 
[Glove 6B](https://nlp.stanford.edu/projects/glove/) with size 300 were tested. 
For running the code create a folder named **embeddings** in the root directory and put the glove.6B.300d.txt file in it.


Alongside the model notebooks, **src/test** repository, includes notebooks containing dataset and 
pipeline statistics (dataset_stat and pipeline_stat), as well as a notebook showcasing the 
clustering analysis performed using BERTopic.

Finally the **src/test/unit_testing** folder contains python scripts for unit testing of the pipeline 
and pipeline's functions.