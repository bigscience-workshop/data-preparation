import os
import pprint as pp
from collections import OrderedDict, defaultdict
from pathlib import Path
import diff_viewer
import pandas as pd
import streamlit as st
from datasets import load_from_disk, load_dataset

DATASET_DIR_PATH_BEFORE_CLEAN_SELECT = os.getenv("DATASET_DIR_PATH_BEFORE_CLEAN_SELECT")
DATASET_DIR_PATH_PII = os.getenv("DATASET_DIR_PATH_PII")
OPERATION_TYPES = [
    "Applied filter",
    "Applied deduplication function",
    "Applied map function",
]
MAX_LEN_DS_CHECKS = os.getenv("MAX_LEN_DS_CHECKS")


def get_ds(ds_path):
    if ds_path.endswith(".jsonl"):
        ds_path = Path(ds_path)
        ds = load_dataset(str(ds_path.parent), data_files=[str(ds_path.name)], split="train")
    else:
        ds = load_from_disk(ds_path)
    return ds


def next_idx(idx: int):
    idx += 1
    return idx % len(st.session_state["ds"])


def previous_idx(idx: int):
    idx -= 1
    return idx % len(st.session_state["ds"])


def on_click_next():
    st.session_state["idx_1"] = next_idx(st.session_state["idx_1"])
    st.session_state["idx_2"] = next_idx(st.session_state["idx_2"])


def on_click_previous():
    st.session_state["idx_1"] = previous_idx(st.session_state["idx_1"])
    st.session_state["idx_2"] = previous_idx(st.session_state["idx_2"])


def on_ds_change(ds_path):
    st.session_state["ds"] = get_ds(ds_path)
    st.session_state["idx_1"] = 0
    st.session_state["idx_2"] = 1 if len(st.session_state["ds"]) > 1 else 0
    st.session_state["ds_name"] = ds_path


def get_log_stats_df(raw_log):
    data = OrderedDict(
        {
            "Order": [],
            "Name": [],
            "Initial number of samples": [],
            "Final number of samples": [],
            "Initial size in bytes": [],
            "Final size in bytes": [],
        }
    )

    metric_dict = defaultdict(lambda: {})
    order = 0
    for line in raw_log.split("\n"):
        for metric_name in list(data.keys()) + OPERATION_TYPES:

            if metric_name == "Name" or metric_name == "Order":
                continue

            if metric_name not in line:
                continue

            if (
                metric_name == "Removed percentage"
                and "Removed percentage in bytes" in line
            ):
                continue

            if (
                metric_name == "Deduplicated percentage"
                and "Deduplicated percentage in bytes" in line
            ):
                continue

            value = line.split(metric_name)[1].split(" ")[1]

            if metric_name in OPERATION_TYPES:
                operation_name = value
                metric_dict[operation_name]["Order"] = order
                order += 1
                continue

            assert (
                metric_name not in metric_dict[operation_name]
            ), f"operation_name: {operation_name}\n\nvalue: {value}\n\nmetric_dict: {pp.pformat(metric_dict)} \n\nmetric_name: {metric_name} \n\nline: {line}"
            metric_dict[operation_name][metric_name] = value
    for name, data_dict in metric_dict.items():
        for metric_name in data.keys():
            if metric_name == "Name":
                data[metric_name].append(name)
                continue

            data[metric_name].append(data_dict[metric_name])
    df = pd.DataFrame(data)
    df.rename(
        {
            "Initial size in bytes": "Initial size (GB)",
            "Final size in bytes": "Final size (GB)",
        },
        axis=1,
        inplace=True,
    )
    df["% samples removed"] = (
        (
            df["Initial number of samples"].astype(float)
            - df["Final number of samples"].astype(float)
        )
        / df["Initial number of samples"].astype(float)
        * 100
    )
    df["Size (GB) % removed"] = (
        (df["Initial size (GB)"].astype(float) - df["Final size (GB)"].astype(float))
        / df["Initial size (GB)"].astype(float)
        * 100
    )
    return df


def get_logs_stats(log_path):
    with open(log_path) as f:
        raw_log = f.read()

    try:
        df = get_log_stats_df(raw_log)
        st.dataframe(df)
    except Exception as e:
        st.write(e)
        st.write("Subset of the logs:")
        subcontent = [
            line
            for line in raw_log.split("\n")
            if "INFO - __main__" in line
            and "Examples of" not in line
            and "Examples n°" not in line
        ]
        st.write(subcontent)


def meta_component(idx_key: str = "idx_1"):
    if "meta" not in st.session_state["ds"][st.session_state[idx_key]]:
        return

    with st.expander("See meta field of the example"):
        meta = st.session_state["ds"][st.session_state["idx_1"]]["meta"]
        st.write(meta)


def filter_page():
    idx_1 = st.session_state["idx_1"]
    idx_2 = st.session_state["idx_2"]
    text_1 = st.session_state["ds"][idx_1]["text"]
    text_2 = st.session_state["ds"][idx_2]["text"]

    st.markdown(
        f"<h1 style='text-align: center'>Some examples of filtered out texts</h1>",
        unsafe_allow_html=True,
    )
    col_button_previous, _, col_button_next = st.columns(3)

    col_button_next.button(
        "Go to next example",
        key=None,
        help=None,
        on_click=on_click_next,
        args=None,
        kwargs=None,
    )
    col_button_previous.button(
        "Go to previous example",
        key=None,
        help=None,
        on_click=on_click_previous,
        args=None,
        kwargs=None,
    )
    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader(f"Example n°{idx_1}")
        meta_component(idx_key="idx_1")
        text_1_show = text_1.replace("\n", "<br>")
        st.markdown(f"<div>{text_1_show}</div>", unsafe_allow_html=True)

    with col_2:
        st.subheader(f"Example n°{idx_2}")
        meta_component(idx_key="idx_2")
        text_2_show = text_2.replace("\n", "<br>")
        st.markdown(f"<div>{text_2_show}</div>", unsafe_allow_html=True)


def dedup_or_cleaning_page():
    col_button_previous, col_to_idx, col_button_next = st.columns((1,6,1))

    st.session_state["idx_1"] = col_to_idx.slider('Current idx', 0, len(st.session_state["ds"]), st.session_state["idx_1"])

    st.markdown(
        f"<h1 style='text-align: center'>Example n°{st.session_state['idx_1']}</h1>",
        unsafe_allow_html=True,
    )
    col_button_next.button(
        "Go to next example",
        key=None,
        help=None,
        on_click=on_click_next,
        args=None,
        kwargs=None,
    )
    col_button_previous.button(
        "Go to previous example",
        key=None,
        help=None,
        on_click=on_click_previous,
        args=None,
        kwargs=None,
    )

    text = st.session_state["ds"][st.session_state["idx_1"]]["text"]
    old_text = st.session_state["ds"][st.session_state["idx_1"]]["old_text"]
    st.markdown(
        f"<h2 style='text-align: center'>Changes applied</h1>", unsafe_allow_html=True
    )
    col_text_1, col_text_2 = st.columns(2)
    with col_text_1:
        st.subheader("Old text")
    with col_text_2:
        st.subheader("New text")
    diff_viewer.diff_viewer(old_text=old_text, new_text=text, lang="none")
    meta_component(idx_key="idx_1")

    with st.expander("See full old and new texts of the example"):
        text_show = text.replace("\n", "<br>")
        old_text_show = old_text.replace("\n", "<br>")

        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader("Old text")
            st.markdown(f"<div>{old_text_show}</div>", unsafe_allow_html=True)
        with col_2:
            st.subheader("New text")
            st.markdown(f"<div>{text_show}</div>", unsafe_allow_html=True)

def page_cleaning_pipeline():
    st.write(
        "The purpose of this application is to sequentially view the changes made to a dataset."
    )
    col_option_clean, col_option_ds = st.columns(2)

    CLEANING_VERSIONS = sorted(list(os.listdir(DATASET_DIR_PATH_BEFORE_CLEAN_SELECT)), reverse=True)
    option_clean = col_option_clean.selectbox(
        "Select the cleaning version", CLEANING_VERSIONS
    )

    DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH_BEFORE_CLEAN_SELECT, option_clean)
    dataset_names = sorted(list(os.listdir(DATASET_DIR_PATH)))
    option_ds = col_option_ds.selectbox("Select the dataset", dataset_names)

    checks_path = os.path.join(DATASET_DIR_PATH, option_ds, "checks")
    checks_names = sorted(list(os.listdir(checks_path)))

    log_path = os.path.join(DATASET_DIR_PATH, option_ds, "logs.txt")
    get_logs_stats(log_path=log_path)

    option_check = st.selectbox("Select the operation applied to inspect", checks_names)
    ds_path = os.path.join(checks_path, option_check)

    if "ds" not in st.session_state or ds_path != st.session_state["ds_name"]:
        on_ds_change(ds_path)

    if len(st.session_state["ds"]) == MAX_LEN_DS_CHECKS:
        st.warning(
            f"Note: only a subset of size {MAX_LEN_DS_CHECKS} of the modified / filtered examples can be shown in this application"
        )
    with st.expander("See details of the available checks"):
        st.write(st.session_state["ds"])


    _ = filter_page() if "_filter_" in option_check else dedup_or_cleaning_page()

def page_pii():
    DATASET_DIR_PATH = DATASET_DIR_PATH_PII
    dataset_names = sorted(list(os.listdir(DATASET_DIR_PATH)))
    option_ds = st.selectbox("Select the dataset", dataset_names)
    ds_path = os.path.join(DATASET_DIR_PATH, option_ds)

    if "ds" not in st.session_state or ds_path != st.session_state["ds_name"]:
        on_ds_change(ds_path)

    if len(st.session_state["ds"]) == MAX_LEN_DS_CHECKS:
        st.warning(
            f"Note: only a subset of size {MAX_LEN_DS_CHECKS} of the modified / filtered examples can be shown in this application"
        )
    with st.expander("See details of the available checks"):
        st.write(st.session_state["ds"])


    dedup_or_cleaning_page()

# Streamlit page
st.set_page_config(page_title="Dataset explorer", page_icon=":hugging_face:", layout="wide")

PAGES = ["none", "cleaning pipeline", "PII"]

option_page = st.selectbox("Which logs do you want to see", PAGES)

if option_page == "cleaning pipeline":
    page_cleaning_pipeline()
elif option_page == "PII":
    page_pii()