C<sup>3</sup>
=====
Overview
--------
This repository maintains **C<sup>3</sup>**, the first free-form multiple-**C**hoice **C**hinese machine reading **C**omprehension dataset.

* Paper: https://arxiv.org/abs/1904.09679
```
@article{sun2019investigating,
  title={Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension},
  author={Sun, Kai and Yu, Dian and Yu, Dong and Cardie, Claire},
  journal={Transactions of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/1904.09679v3}
}
```

Files in this repository:

* ```license.txt```: the license of C<sup>3</sup>.
* ```data/c3-{m,d}-{train,dev,test}.json```: the dataset files, where m and d represent "**m**ixed-genre" and "**d**ialogue", respectively. The data format is as follows.
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
      },
      {
        "question": document 1 / question 2,
        "choice": [
          document 1 / question 2 / answer option 1,
          document 1 / question 2 / answer option 2,
          ...
        ],
        "answer": document 1 / question 2 / correct answer option
      },
      ...
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
      },
      {
        "question": document 2 / question 2,
        "choice": [
          document 2 / question 2 / answer option 1,
          document 2 / question 2 / answer option 2,
          ...
        ],
        "answer": document 2 / question 2 / correct answer option
      },
      ...
    ],
    document 2 / id
  ],
  ...
]
```
* ```annotation/c3-{m,d}-{dev,test}.txt```: question type annotations. Each file contains 150 annotated instances. We adopt the following abbreviations:


<table>
  <tr>
    <th></th>
    <th>Abbreviation</th>
    <th>Question Type</th>
  </tr>
  <tr>
    <td rowspan="1">Matching</td>
    <td>m</td>
    <td>Matching</td>
  </tr>
  <tr>
    <td rowspan="10">Prior knowledge</td>
    <td>l</td>
    <td>Linguistic</td>
  </tr>
  <tr>
    <td>s</td>
    <td>Domain-specific</td>
  </tr>
  <tr>
    <td>c-a</td>
    <td>Arithmetic</td>
  </tr>
  <tr>
    <td>c-o</td>
    <td>Connotation</td>
  </tr>
  <tr>
    <td>c-e</td>
    <td>Cause-effect</td>
  </tr>
  <tr>
    <td>c-i</td>
    <td>Implication</td>
  </tr>
  <tr>
    <td>c-p</td>
    <td>Part-whole</td>
  </tr>
  <tr>
    <td>c-d</td>
    <td>Precondition</td>
  </tr>
  <tr>
    <td>c-h</td>
    <td>Scenario</td>
  </tr>
  <tr>
    <td>c-n</td>
    <td>Other</td>
  </tr>
  <tr>
    <td rowspan="3">Supporting Sentences</td>
    <td>0</td>
    <td>Single Sentence</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Multiple sentences</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Independent</td>
  </tr>
</table>
