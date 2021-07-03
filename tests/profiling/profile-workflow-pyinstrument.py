# type: ignore
# pylint: disable-all
# isort: skip

import sys

from workflow import run_workflow as workflow

sys.path.append('./tests/profiling')


if __name__ == '__main__':

    workflow()
