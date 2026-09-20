# ===========================================================
# main.py — Entry point cho dự án KMeans Routing in UWSNs
# ===========================================================

from uwsn.runner import main
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
