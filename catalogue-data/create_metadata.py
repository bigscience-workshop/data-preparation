import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from datasets import load_dataset
from datasets.commands.test import TestCommand
from huggingface_hub import HfApi, Repository


HF_ORG = "bigscience-catalogue-lm-data"
DATASETS_DIR = "datasets"


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="language code, as 'ca' or 'indic' or 'nigercongo'")
    parser.add_argument("--repo", action="append", help="repo ID")
    parser.add_argument("--force", action="store_true", help="force re-creation of metadata file")
    args = parser.parse_args()
    return args


def init():
    # Env variables
    if Path(".env").exists:
        load_dotenv()
    # Login
    api = HfApi()
    api.set_access_token(os.environ["HF_USER_ACCESS_TOKEN"])
    # Filesystem
    os.makedirs(DATASETS_DIR, exist_ok=True)
    return api


# TODO
def list_repos(lang=None, api=None):
    repos = api.list_datasets(author=HF_ORG)  # TODO: only public datasets are listed
    return repos


def main(lang=None, repo=None, force=False):
    api = init()
    if lang:
        repos = list_repos(lang=lang, api=api)
    elif repo:
        repos = repo
    if repos:
        for repo_id in repos:
            logger.info(f"Start: {repo_id}")
            repo_name = repo_id.split("/")[-1]
            local_dir = f"{DATASETS_DIR}/{repo_name}"
            repo = Repository(
                local_dir=local_dir,
                clone_from=repo_id,
                repo_type="dataset",
                use_auth_token=os.environ["HF_USER_ACCESS_TOKEN"],
                git_user=os.environ["GIT_USER"],
                git_email=os.environ["GIT_EMAIL"],
            )
            repo.git_pull()
            if os.path.exists(local_dir + "/dataset_infos.json") and not force:
                continue
            # First, load dataset to fix auth problems generating metadata
            _ = load_dataset(local_dir, use_auth_token=os.environ["HF_USER_ACCESS_TOKEN"], ignore_verifications=True)
            # Then, generate metadata
            metadata_args = {
                "dataset": local_dir,
                "name": None,
                "cache_dir": None,
                "data_dir": None,
                "all_configs": True,
                "save_infos": True,
                "ignore_verifications": False,
                "force_redownload": False,
                "clear_cache": False,
                "proc_rank": 0,
                "num_proc": 1,
            }
            metadata_command = TestCommand(*metadata_args.values())
            metadata_command.run()
            # Commit
            repo.git_add()
            repo.git_commit(commit_message="Add metadata")
            repo.git_push()
            logger.info(f"End: {repo_id}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
