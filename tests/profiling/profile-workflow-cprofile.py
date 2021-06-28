# type: ignore
# pylint: disable-all
# isort: skip

import cProfile
import pstats
import sys

from workflow import run_workflow

sys.path.append('./tests/profiling')




def workflow():
    profiler = cProfile.Profile()
    profiler.enable()
    run_workflow()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()


if __name__ == '__main__':

    cProfile.run('workflow()')
