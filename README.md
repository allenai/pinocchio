# The Pinocchio Algorithm  

This Repo contains the source code for [Don't Say What You Don't Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search](https://arxiv.org/abs/2203.08436).

## Overview 


## Running Pinocchio 

### Minimal starter code for the original implementation in transformers 2

#### Installation 

```bash
conda create -n pinocchio python=3.7
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
It will install a customized version of [HuggingFace transformers](https://github.com/dakinggg/transformers/tree/my_branch) with some edits to the beam search code. 

#### Use Pinocchio with BART on XSUM 

```bash
cd piocchio/ # the root dir of this repo 

# you might want to make sure there's one GPU for running this  
python example.py output.json --gpu_id 0
```
Right now we only support running with BART. 

#### Explanation of the code 

We change the [beam search decoding process](https://github.com/dakinggg/transformers/blob/defbd001072dffbedf03c7f9eb880f2836c640e3/src/transformers/modeling_utils.py#L1333) by 
1. monitoring additional metrics like token entropy / attribution for each generation step  
2. adding a customized `beam_search_scorer` and other changes to the decoding process for the constrained beam search algorithm 

And the `generic_text_predictions` function uses the updated beam search decoding code and parse the outputs. 

### Adapting the code to transformers>4

[WIP]

## Data Releases 

[WIP]

## Cite Our Work 

```
@article{king2022don,
  title={Don't Say What You Don't Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search},
  author={King, Daniel and Shen, Zejiang and Subramani, Nishant and Weld, Daniel S and Beltagy, Iz and Downey, Doug},
  journal={arXiv preprint arXiv:2203.08436},
  year={2022}
}
```