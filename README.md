
Improving Machine Reading Comprehension with Contextualized Commonsense Knowledge
=====

This repository maintains the code and resource for the above ACL'22 paper. Please contact script@dataset.org if you have any questions or suggestions.

* Paper: https://arxiv.org/abs/2009.05831v2
```
@inproceedings{sun2022improving,
  title={Improving Machine Reading Comprehension with Contextualized Commonsense Knowledge},
  author={Sun, Kai, Yu, Dian, Chen, Jianshu, and Yu, Dong and Cardie, Claire},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022},
  url={https://arxiv.org/abs/2009.05831v2}
```

Files in this repository:

* ```license.txt```: the license of the code and data.
* ```data/lb/{lb1,lb2}.json```: samples of the weakly-labeled MRC instances constructed by xx (used in the paper). 
* ```data/gb/{lb1,lb2}.json```: samples of the weakly-labeled MRC instances constructed by xx (used in the paper). 
* ```data/ctb/{lb1,lb2}.json```: samples of the weakly-labeled MRC instances constructed by xx (used in the paper). 
* ```data/ib/{lb1,lb2}.json```: samples of the weakly-labeled MRC instances constructed by xx (used in the paper). 


The data format is as follows.
```
[
  [
    [
      document 1
    ],
    [
      {
        "question": document 1 / question 1,
        "choice": [
          document 1 / question 1 / answer option 1,
          document 1 / question 1 / answer option 2,
          ...
        ],
        "answer": document 1 / question 1 / correct answer option
      }
    ],
    document 1 / id
  ],
  [
    [
      document 2
    ],
    [
      {
        "question": document 2 / question 1,
        "choice": [
          document 2 / question 1 / answer option 1,
          document 2 / question 1 / answer option 2,
          ...
        ],
        "answer": document 2 / question 1 / correct answer option
      }
    ],
    document 2 / id
  ],
  ...
]
```

**Experiment**:
=====

**STEP1: Train Teacher Models**

Set the file paths for the pre-trained language model, c3, and four sets of weakly-labeled data based on contextualized knowledge in ```run-roberta-wwm-ext-large-teacher.sh```and execute
```bash run-roberta-wwm-ext-large-teacher.sh```.

**STEP2: Generate Soft Lables for Both Weakly-Labled and Clean Data**


**STEP3: Train a Student Model**


* ```data_v2/en/data/{train,dev,test}.json```: the updated dataset files with a few annotation errors fixed. The format is the same as the orignal. (**Updated on Aug 2020**)
* ```data_v2/cn/data/{train,dev,test}.json```: a Chinese version of DialogRE. The format is the same as the orignal. Please note that since ground truth argument types do not substantially contribute to the performance according to Section 5.2 of our paper, we no longer annotate argument types when annotating the Chinese version. Instead, all ```"x_type"```s and ```"y_type"```s are left empty. (**Updated on Aug 2020**)
* ```kb/Fandom_triples```: relational triples from [Fandom](https://friends.fandom.com/wiki/Friends_Wiki).
* ```kb/matching_table.txt```: mapping from Fandom relational types to DialogRE relation types.
* ```bert``` folder: a re-implementation of BERT and BERT<sub>S</sub> baselines.
  1. Download and unzip BERT from [here](https://github.com/google-research/bert), and set up the environment variable for BERT by 
  ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
  2. Copy the dataset folder ```data``` (or ```data_v2/{en,cn}/data``` for the updated version) to ```bert/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.
  4. To run and evaluate the BERT baseline, execute the following commands in ```bert```:
  ```
  python run_classifier.py   --task_name bert  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir bert_f1  --gradient_accumulation_steps 2
  rm bert_f1/model_best.pt && cp -r bert_f1 bert_f1c && python run_classifier.py   --task_name bertf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir bert_f1c  --gradient_accumulation_steps 2
  python evaluate.py --f1dev bert_f1/logits_dev.txt --f1test bert_f1/logits_test.txt --f1cdev bert_f1c/logits_dev.txt --f1ctest bert_f1c/logits_test.txt
  ```
  5. To run and evaluate the BERT<sub>S</sub> baseline, execute the following commands in ```bert```:
  ```
  python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 2
  rm berts_f1/model_best.pt && cp -r berts_f1 berts_f1c && python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 2
  python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt
  ```
**Environment**:
  The code has been tested with Python 3.6 and PyTorch 1.0.

**TODO**:

- [x] Release DialogRE
- [x] Release a Chinese version of DialogRE (summer 2020)
- [x] Fix the annotation errors in DialogRE and release an updated English version (summer 2020) 
- [x] Baseline results for the updated version ([here](https://dataset.org/dialogre/))
