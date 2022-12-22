import zipfile

import click
import pandas as pd

import penelope.corpus.readers as pr
import penelope.pipeline as pp


@click.command()
@click.argument('source_filename', type=click.STRING)
@click.argument('config_filename', type=click.STRING)
@click.argument('target_filename', type=click.STRING)
@click.option('--reduce-key', type=click.STRING, default='year')
def reduce_corpus(source_filename: str, config_filename: str, target_filename: str, reduce_key: str = 'year'):
    reduce_by_key(source_filename, config_filename, target_filename, reduce_key)


def reduce_by_key(source_filename: str, config_filename: str, target_filename: str, reduce_key: str):

    corpus_config: pp.CorpusConfig = pp.CorpusConfig.load(config_filename)
    reader_opts = corpus_config.text_reader_opts
    reader: pr.TextReader = pr.TextReader(source_filename, reader_opts=reader_opts)
    di: pd.DataFrame = reader.document_index

    if reduce_key not in di.columns:
        raise ValueError(f"reduce key '{reduce_key}' not found in document index")

    reduce_keys = sorted(list(di[reduce_key].unique()))
    di_str: str = f"document_id\tfilename\tdocument_name{reduce_key}\tn_raw_tokens\n"

    with zipfile.ZipFile(target_filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:

        for i, key in enumerate(reduce_keys):

            reader_opts.filename_filter = sorted(list(di[di[reduce_key] == key].filename))
            key_reader = pr.TextReader(source_filename, reader_opts=reader_opts)

            document_name: str = f"{key}_{i:04}_9999"
            if key != 'year':
                """Add year to document name"""
                years: list[int] = key_reader.document_index.year.unique().tolist()
                if len(years) == 1:
                    document_name = f"{years[0]}_{document_name}"

            document_filename: str = f"{document_name}.txt"

            document_str: str = '\n'.join(t for _, t in key_reader)

            di_str += f"{i}\t{document_filename}\t{document_name}\t{key}\t{len(document_str.split())}\n"

            zf.writestr(document_filename, data=document_str)

        zf.writestr("document_index.csv", data=di_str)


if __name__ == "__main__":
    reduce_corpus()  # pylint: disable=no-value-for-parameter

    # from click.testing import CliRunner

    # runner = CliRunner()
    # result = runner.invoke(
    #     reduce_corpus,
    #     [
    #         '/data/inidun/courier/courier_page_20210921.zip',
    #         '/home/roger/source/penelope/opts/inidun/20221214_co_occurrence/courier/courier_page.yml',
    #         'apa.zip'
    #     ],
    # )
    # print(result.output)
