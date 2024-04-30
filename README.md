# CVulKeys
The Model of CVulKeys

## Description
***
model.py contains models.

train.py is used to train.
***

## Data
Download the data used in paper from [here](https://drive.google.com/drive/folders/1dxqaMP-9YSoPrCK-ymnHfTkiDkZLQ-VE?usp=drive_link).

***
*Put word2vec model in /model* 

#  Getting Started



This documentation helps you run the code on your own machine

## Prepare

Install the necessary dependencies before running the project

### software

joern 

python 3.8.18

gensim 3.8.3

networkx 2.6.3

nltk 3.8.1

numpy 1.23.5

pydot 1.4.2

scikit-learn

# Keyword Extraction

use TF-IDF generate keyword_list.txt

# Graph Extraction

use joern generate PDG 

we apply joern-cli to generate PDG

run joern_graph.py 

[Joern_graph.py]: https://drive.google.com/file/d/13VoKA6fRdTQU-cV-0DNhFBeY8wp82-dH/view?usp=share_link

You can use  the data in /data 

[Data]: https://drive.google.com/file/d/15KPqpc6AtjqTicYCKwHuXhusuucFDkvi/view?usp=share_link



## generate .bin files

```
python joern_graph.py  -i ./data/Vul -o ./data/bins/Vul -t parse
python joern_graph.py  -i ./data/No-Vul -o ./data/bins/No-Vul -t parse

```

## generate pdg(.dot files)

```
python joern_graph.py  -i ./data/bins/Vul -o ./data/pdg/Vul -t export -r pdg
python joern_graph.py  -i ./data/bins/Vul -o ./data/pdg/No-Vul -t export -r pdg
```

After graph extraction, data is generate into pdg 

[pdg]: https://drive.google.com/file/d/1sO_ucOMsA7klN19XlSH6RICCjepg_YAu/view?usp=share_link



# Graph Extraction & Feature Extraction

got the processing_back folder 

[Process_back folder]: https://drive.google.com/drive/folders/1TeFoz6wbgrK8A1tmvRSOhHgij4ZQpDNt?usp=share_link

find_root_line.py contains basic function of slice and find_root_line.

slice.py is used to slice the .dot file in a folder.

Keyword_list.txt is the keyword file

## slice pdg

```
python slice.py
```

folder = "../NewVul_Model\pdg" 

The first level directory in the folder must be the category name. For example, in the picture below, the folder is pdg, and the first-level directory is the category.

```
pdg - No_Vul    ----

              - Vul_CWE1  ----

              - Vul_CWE2 ----

               ...
          
               - Vul_CWE6  -----
```

## build pickle

```
python build_pickle.py
```

After this step, the all_data.pickle file is generated in /process_pdg folder

[Process_pdg]: https://drive.google.com/file/d/1ixxrLQ8qdU4QC88z6uO6KbZJ9s044LsT/view?usp=share_link



# Embedding

using word2vec in model/middle_word2vec.pkl

[word2vec]: https://drive.google.com/drive/folders/19Gu0jJDe1iwZeG50unaqqIuWmWKm5qUz?usp=share_link

If you want to achieve better performance of CVulKey, you'd better train a new word2vec by using larger dataset 

# Train with CNN

using word2vec in model/middle_word2vec.pkl

using all_data.pickle  in /process_pdg folder

```
python train.py
```


