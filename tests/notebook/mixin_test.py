from penelope.notebook.mixins import PivotKeysMixIn


def test_pivot_keys_mixin():
    pivot_keys: dict = {
        'fågel': {
            'text_name': 'fågel',
            'id_name': 'fågel_id',
            'values': {
                'okänd': 0,
                'kråka': 1,
                'skata': 2,
            },
        },
        'husdjur': {
            'text_name': 'husdjur',
            'id_name': 'husdjur_id',
            'values': {
                'okänd': 0,
                'katt': 1,
                'hund': 2,
            },
        },
    }
    gui = PivotKeysMixIn()

    gui.pivot_keys = pivot_keys
