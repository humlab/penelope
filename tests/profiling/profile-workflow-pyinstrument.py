# type: ignore
# pylint: disable-all
# isort: skip

import cProfile
import pstats
import sys

from workflow import run_workflow as workflow

sys.path.append('./tests/profiling')


if __name__ == '__main__':

    cProfile.run('workflow()')
