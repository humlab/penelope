
tokenized_corpus_co_occurrence:
	pyinstrument -o ./tokenized_corpus_co_occurrence.log tokenized_corpus_co_occurrence.py

pipeline_co_occurrence:
	pyinstrument -o ./pipeline_co_occurrence.log pipeline_co_occurrence.py

# Usage: pyinstrument [options] scriptfile [arg] ...
# Options:
#   --version             show program's version number and exit
#   -h, --help            show this help message and exit
#   --load-prev=ID        Instead of running a script, load a previous report
#   -m MODULE_NAME        run library module as a script, like 'python -m
#                         module'
#   -o OUTFILE, --outfile=OUTFILE
#                         save to <outfile>
#   -r RENDERER, --renderer=RENDERER
#                         how the report should be rendered. One of: 'text',
#                         'html', 'json', or python import path to a renderer
#                         class
#   -t, --timeline        render as a timeline - preserve ordering and don't
#                         condense repeated calls
#   --hide=EXPR           glob-style pattern matching the file paths whose
#                         frames to hide. Defaults to '*/lib/*'.
#   --hide-regex=REGEX    regex matching the file paths whose frames to hide.
#                         Useful if --hide doesn't give enough control.
#   --show=EXPR           glob-style pattern matching the file paths whose
#                         frames to show, regardless of --hide or --hide-regex.
#                         For example, use --show '*/<library>/*' to show frames
#                         within a library that would otherwise be hidden.
#   --show-regex=REGEX    regex matching the file paths whose frames to always
#                         show. Useful if --show doesn't give enough control.
#   --show-all            show everything
#   --unicode             (text renderer only) force unicode text output
#   --no-unicode          (text renderer only) force ascii text output
#   --color               (text renderer only) force ansi color text output
#   --no-color            (text renderer only) force no color text output
