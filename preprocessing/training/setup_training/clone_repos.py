from pathlib import Path

from huggingface_hub import Repository

"""Script used to clone repos, as it's needed in order to load data stored in the hub in OFFLINE mode. Thanks leandro"""

DATASETS_DIR=Path("/gpfswork/rech/six/commun/train_tokenizer/datasets")

repos = [
    "bigscience-catalogue-lm-data/lm_id_indonesian_news_articles_2017",
    "bigscience-catalogue-lm-data/tokenization_nigercongo",
    "bigscience-catalogue-lm-data/lm_code_github",
    "bigscience-catalogue-lm-data/lm_ar_wikipedia",
    "bigscience-catalogue-lm-data/lm_ca_wikipedia",
    "bigscience-catalogue-lm-data/lm_en_wikipedia",
    "bigscience-catalogue-lm-data/lm_es_wikipedia",
    "bigscience-catalogue-lm-data/lm_eu_wikipedia",
    "bigscience-catalogue-lm-data/lm_fr_wikipedia",
    "bigscience-catalogue-lm-data/lm_id_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-as_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-bn_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-gu_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-hi_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-kn_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-ml_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-mr_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-or_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-pa_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-ta_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-te_wikipedia",
    "bigscience-catalogue-lm-data/lm_indic-ur_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ak_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-bm_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ig_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ki_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-lg_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ln_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-nso_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ny_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-rn_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-rw_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-sn_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-st_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-sw_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-tn_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-ts_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-tum_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-tw_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-wo_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-yo_wikipedia",
    "bigscience-catalogue-lm-data/lm_nigercongo-zu_wikipedia",
    "bigscience-catalogue-lm-data/lm_pt_wikipedia",
    "bigscience-catalogue-lm-data/lm_vi_wikipedia",
    "bigscience-catalogue-lm-data/lm_zh-cn_wikipedia",
    "bigscience-catalogue-lm-data/lm_zh-tw_wikipedia",
]

for repo_id in repos:
    repo_name = repo_id
    (DATASETS_DIR / repo_id).parent.mkdir(parents=True, exist_ok=True)
    repo = Repository(
                local_dir=f"{DATASETS_DIR}/{repo_name}",
                clone_from=repo_id,
                repo_type="dataset",
                use_auth_token=True,
                # git_user=os.environ["GIT_USER"],
                # git_email=os.environ["GIT_EMAIL"],
    )
    repo.git_pull()
