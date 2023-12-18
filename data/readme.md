Here the folder with datasets. There are 3 types : 

##### psychology-char folder
It's a dataset with Q&A of psychology, which will be splited in char in ``config/train_psychology.py``.

##### psychology folder
This is the dataset use in ``config/finetune_psychology.py``. It's normally better than the first, but heavier. It can be finetuned based on gpt2 weights.

##### not-nice folder
It's a dataset of bad Q&A. It can be use if you want to create an evil psychology chatbot.


##### ``data.ipynb`` 
It's a notebook to show how to create .txt file with Q&A.