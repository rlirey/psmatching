import psmatch_utilities
import glob
import time
from dask.diagnostics import ProgressBar
from dask import compute, delayed


if __name__ == "__main__":

    start = time.time()

    pBar = ProgressBar()
    pBar.register()

    print()
    model = input("Enter model for obtaining propensity scores: ")
    path = input("Please enter path for data folder: ")
    files = glob.glob(path + "*].csv")
    files = sorted(files)

    compute([delayed(run)(file) for file in files], scheduler = "processes", num_workers = 3)

    end = time.time()
    elapsed = end-start
    print("\nTime elapsed: %d minutes" % (elapsed/60))
