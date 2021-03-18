
from tests.pipeline.parlaclarin.parlaclarin_xml_to_text import ParlaClarinXml2Csv


def test_parla_clarin_to_csv():

    # test_filename = './tests/test_data/prot-1956-höst-fk--26.xml'
    test_filename = './tests/test_data/prot-199394--124.xml'

    parser = ParlaClarinXml2Csv()

    csv_str = parser.read_transform(test_filename)

    assert csv_str is not None

    with open('./tests/output/parla_clarin.csv', 'w') as fp:
        fp.write(csv_str)
