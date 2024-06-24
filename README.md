# EGRec

EGRec is an anonymous link to our paper submission "Leveraging Generative Implicit Intents for Enhanced Recommendation with Large Language Models".



## Requirements
```
python>=3.8
transformers>=4.39.0
numpy
pytorch>=1.10.0
scikit-learn
transformers_stream_generator
einops
tiktoken
```

## Setup

1. Download dataset
   
   Take Amazon-Books for example, download the dataset to folder `data/amz/raw_data/`
2. Preprocessing: in folder `preprocess`
   1. run `python preprocess_amz.py`.
   2. run `generate_data.py` to generate data for CTR  task.
   3. Download the prompt file we produced from the geogle drive link. Of course, you can also produce it according to the paper, but it will be slower.
   

3. model: in folder `model`

   Run `python main.py` ,you can choose the backbone yourself.

 
