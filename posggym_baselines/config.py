import os
import os.path as osp

PKG_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.abspath(osp.join(PKG_DIR, os.pardir))
BASE_RESULTS_DIR = osp.join(REPO_DIR, "results")


if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)
