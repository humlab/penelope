## Inferred Topics Data Bundle Description

### Overview

These files are the minimal data set needed for exploring a topic model using Westac's topic ,odeling Jupyter notebooks. See example test bundle in [tests/test_data/tranströmer_inferred_model](https://github.com/humlab/penelope/tree/main/tests/test_data/transtr%C3%B6mer/transtr%C3%B6mer_inferred_model).

Note that the "CSV files" in actuality are tab-separated files. All CSV files can be stored  [feather](https://arrow.apache.org/docs/python/feather.html) files (highly recommended for large corpora).

| File                   | Description                             | Type     |
| ---------------------- | --------------------------------------- | -------- |
| dictionary             | Vocabulary with unique token identities | CSV, ZIP |
| documents              | Index of corpus' documents              | CSV, ZIP |
| document_topic_weights | Topics' distribution over documents     | CSV, ZIP |
| topic_token_weights    | Token's distribution over topics        | CSV, ZIP |
| topic_tokens_overview  | Topic token overview & topic labelling  | CSV, ZIP |

If the bundle is created with `tm-predict@penelope` then  run-time options are stored in JSON file `model_options.json`.


### Dictionary

File `dictionary.[csv,zip]` contains `token` to `token id` mapping.

| Column   | Type | Description                            | Mandatory |
| -------- | ---- | -------------------------------------- | --------- |
| token_id | int  | Token's unique identity                | yes       |
| token    | str  | Token / word type                      | yes       |
| dfs      | int  | Number of documents where token occurs | no        |

Note that `token_id` must be a strictly increasing unique integer for each `token`.

### Documents

File `documents.[csv,zip]` contains an index of all the documents. Note that `document_id` must be a strictly increasing unique integer.

| Column        | Type | Description                                          | Mandatory |
| ------------- | ---- | ---------------------------------------------------- | --------- |
| filename      | str  | Document's filename without path                     | yes       |
| document_id   | int  | A strictly increasing (starting from 0) integer ID   | yes       |
| document_name | str  | Document's name (usually filename without extension) | yes       |
| year          | int  | Document's year                                      | yes       |
| n_tokens      | int  | Number of tokens in document                         | yes       |
| ...           | ...  | Any additional document specific attributes          | no        |

Example:

| filename              | year | year_serial_id | document_id | document_name     | title          | n_tokens |
| --------------------- | ---- | -------------- | ----------- | ----------------- | -------------- | -------- |
| tran_2019_01_test.txt | 2019 | 1              | 0           | tran_2019_01_test | Romanska bågar | 68       |
| tran_2019_02_test.txt | 2019 | 2              | 1           | tran_2019_02_test | Nocturne       | 59       |
| tran_2019_03_test.txt | 2019 | 3              | 2           | tran_2019_03_test | Gläntan        | 173      |
| tran_2020_01_test.txt | 2020 | 1              | 3           | tran_2020_01_test | Ostinato       | 33       |
| tran_2020_02_test.txt | 2020 | 2              | 4           | tran_2020_02_test | Epilog         | 44       |

### Document Topics Weights

File `document_topic_weights.[zip,csv]` contains topic distribution over documents.
The data is in certain use cases overloaded with document's year.

| Column      | Type | Description                  | Mandatory |
| ----------- | ---- | ---------------------------- | --------- |
| document_id | int  | Token / word type            | yes       |
| topic_id    | int  | Token / word type            | yes       |
| weight      | int  | Topic's weight in document   | yes       |
| year        | int  | Document's year (overloaded) | no        |

Example:

| document_id | topic_id | weight                | year |
| ----------- | -------- | --------------------- | ---- |
| 0           | 0        | 0.9916352033615112    | 2019 |
| 0           | 1        | 0.0027700476348400116 | 2019 |
| 0           | 2        | 0.0027892703656107187 | 2019 |
| 0           | 3        | 0.00280552264302969   | 2019 |
| 1           | 0        | 0.0034797682892531157 | 2019 |
| 1           | 1        | 0.0034864393528550863 | 2019 |
| 1           | 2        | 0.003491179784759879  | 2019 |
| 1           | 3        | 0.9895426034927368    | 2019 |
| 2           | 0        | 0.0010345447808504105 | 2019 |
| 2           | 1        | 0.0010488786501809955 | 2019 |
| 2           | 2        | 0.9968704581260681    | 2019 |
| 2           | 3        | 0.001046146615408361  | 2019 |
| 3           | 0        | 0.006863145623356104  | 2020 |
| 3           | 1        | 0.9794552326202393    | 2020 |
| 3           | 2        | 0.006856377702206373  | 2020 |
| 3           | 3        | 0.006825248245149851  | 2020 |
| 4           | 0        | 0.9848133325576782    | 2020 |
| 4           | 1        | 0.005064100027084351  | 2020 |
| 4           | 2        | 0.00506967818364501   | 2020 |
| 4           | 3        | 0.005052865482866764  | 2020 |

### Topic Token Weights

File `topic_token_weights.[zip,csv]` contains token distribution over topics.
The data is in certain use cases overloaded with token in plain text.

| Column   | Type  | Description             | Mandatory |
| -------- | ----- | ----------------------- | --------- |
| topic_id | int   | Token / word type       | yes       |
| token_id | int   | Token's unique identity | yes       |
| weight   | float | Token's weight in topic | yes       |

Example:

| topic_id | token_id | weight               |
| -------- | -------- | -------------------- |
| 0        | 38       | 0.04476533457636833  |
| 0        | 58       | 0.02178177796304226  |
| 0        | 23       | 0.02060970477759838  |
| 0        | 4        | 0.015128441154956818 |
| 0        | 45       | 0.013285051099956036 |
| 1        | 38       | 0.02447959966957569  |
| 1        | 49       | 0.02074943669140339  |
| 1        | 4        | 0.020712170749902725 |
| 1        | 23       | 0.018629901111125946 |
| 1        | 13       | 0.016076957806944847 |
| 2        | 49       | 0.02533087506890297  |
| 2        | 13       | 0.023466676473617554 |
| 2        | 63       | 0.022432737052440643 |
| 2        | 23       | 0.019307153299450874 |
| 2        | 38       | 0.01673712022602558  |
| 3        | 73       | 0.02712307684123516  |
| 3        | 63       | 0.023767853155732155 |
| 3        | 23       | 0.019748181104660034 |
| 3        | 13       | 0.017621515318751335 |
| 3        | 49       | 0.015297766774892807 |

### Topics Overview

The `topic_tokens_overview.[zip,csv]` file gives the top `n` tokens for each topic. It can be reconstructed (apart from the `alpha` value) from data in the other files.

| Column   | Type  | Description                | Mandatory |
| -------- | ----- | -------------------------- | --------- |
| topic_id | int   | Topic's name               | yes       |
| tokens   | str   | Top 'n' tokens in topic    | yes       |
| alpha    | float | Topic's weight in document | yes       |

| topic_id | tokens            | alpha |
| -------- | ----------------- | ----- |
| 0        | och valv i av sig | 0.25  |
| 1        | och som av i en   | 0.25  |
| 2        | som en är i och   | 0.25  |
| 3        | de är i en som    | 0.25  |

### Other Optional Files

These files exists if the bundle hase been created with `penelope`. Some files

| File                  | Description                                   |
| --------------------- | --------------------------------------------- |
| corpus.yml            | Train/predict corpus options                  |
| token_diagnostics.zip | Token diagnostics (if and only if MALLET)     |
| topic models          | Topic model files depending on used TM engine |
| ./mallet              | MALLET ouput files (if MALLET)                |
