# Pickle DTM Exploder

This Python script is used to explode a pickled Document-Term Matrix (DTM) saved with `pandas<2.0.0` into separate files. This is necessary because `pickle.load` with `pandas>=2.0.0` fails when reading a dictionary that contains a DataFrame stored with `pandas<2.0.0`.

**Note:** This script must be run with `pandas<2.0.0`!

## Installation

To install the necessary dependencies for this script, first create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
```

Then, install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

To use the script, set the Python path and run the script with your desired root folder as an argument:

```bash
PYTHONPATH=. python penelope/scripts/explode-dtm-pickle/explode-pickle.py <root_folder>
```

The script will recursively search for pickled DTM files in the specified root folder and explode them into separate files.

The input file should be a pickled DTM file named `<tag>_vectorizer_data.pickle`.

The output files will be:

- A CSV file named `<tag>_document_index.csv.gz`
- A JSON file named `<tag>_token2id.json.gz`

## License

This project is licensed under the terms of the MIT license.
```

You can replace the "License" section with the actual license of your project. If your project doesn't have a license, you might want to consider adding one.

Remember to replace `<root_folder>` with the actual path of the folder that you want to process.