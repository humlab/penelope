import penelope.notebook.ipyaggrid_utility as ipyaggrid_utility
from ipyaggrid import Grid


def display_document_topics_as_grid(df):

    column_defs = [
        {"headerName": "Index", "field": "index", 'hide': True},
        {"headerName": "Document_Id", "field": "document_id"},
        {"headerName": "Topic_Id", "field": "topic_id"},
        {
            "headerName": "Weight",
            "field": "weight",
            "cellRenderer": "function(params) { return params.value.toFixed(6); }",
        },
        {"headerName": "Filename", "field": "filename"},
        {"headerName": "N_Raw_Tokens", "field": "n_raw_tokens"},
        {"headerName": "N_Tokens", "field": "n_tokens"},
        {"headerName": "N_Terms", "field": "n_terms"},
        {"headerName": "Year", "field": "year", "cellRenderer": None},
        {"headerName": "Title", "field": "title"},
    ]

    extra_options = {
        "menu": {
            "buttons": [
                {"name": "Export Grid", "hide": True},
                {
                    'name': 'Export to Gephi',
                    'custom_css': {'background': 'lightgreen'},
                    'action': 'gridOptions.api.exportDataAsCsv({columnKeys: ["topic_id", "filename", "weight"], skipHeader: true, customHeader: ["Source", "Target", "Weight"], fileName: "gephi.csv"});',
                },
            ]
        },
    }
    grid_style = {**ipyaggrid_utility.DEFAULT_GRID_STYLE, **extra_options}

    grid_options = dict(columnDefs=column_defs, **ipyaggrid_utility.DEFAULT_GRID_OPTIONS)
    g = Grid(grid_data=df, grid_options=grid_options, **grid_style)
    return g
