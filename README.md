The project is a GPT like wich give you psychology advices if you suffer of anxiety or other psychological disorders.

It's based on the project from nanoGPT project from Karparthy (https://github.com/karpathy/nanoGPT)

The web part of the project doesn't work


## To train : 
For the basic one :
``$ python train.py config/train_psychology.py``

For the basic one but based on gpt2 : 
``$ python train.py config/finetune_psychology.py``

For the evil advice chat : 
``$ python train.py config/train_not_nice.py``

## To test :
#### In the terminal :
``$ python3.10 sample.py --out_dir=out-psychology-char``

You can replace the out_dir by the others 

#### On web (local)
Open the index.html file in your browser 