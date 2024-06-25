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
   2. run `generate_data.py` to generate data .
   3. You can download the prompt we generated from the link and put it in `data/amz/raw_data/`. Of course, you can also generate your own prompts based on the paper.
   

3. model: in folder `model`

   Run `python main.py` .

 
