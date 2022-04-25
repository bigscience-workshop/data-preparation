import copy
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, urlsplit

import datasets
import hydra
import wandb
from datasets import Dataset, Features, Value, config, load_dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bsmetadata.preprocessing_tools import html_parser
from bsmetadata.train import show_help

logger = logging.getLogger(__name__)


class MetadataPreprocessor:
    """A metadata processor can be used for preprocessing text and adding or extracting metadata information."""

    def __init__(self, col_to_store_metadata: str) -> None:
        self.col_to_store_metadata = col_to_store_metadata
        super().__init__()

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        """Returns a dictionary whose key corresponds to the name of a new column / a column modified by this processor
        and whose value corresponds to the minimal format of this column"""
        pass

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples and add or extract corresponding metadata."""
        pass


class HtmlPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for extracting metadata from html text.

    Specifically, it separates the html text contained in the `col_html`` column into a text and a list of
    HTML metadata containing the tags, their attributes, their location in the text and their relative location to
    each other."""

    def __init__(
        self,
        col_html: str = "doc_html",
        col_to_store_metadata="metadata",
        col_to_store_text="text",
        col_to_store_head="html_head",
        col_to_store_footer="html_footer",
        col_to_store_title="html_title",
    ) -> None:
        self.col_html = col_html
        self.col_to_store_text = col_to_store_text
        self.col_to_store_footer = col_to_store_footer
        self.col_to_store_head = col_to_store_head
        self.col_to_store_title = col_to_store_title
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        features = {
            self.col_to_store_metadata: [
                {
                    "char_end_idx": Value("int64"),
                    "char_start_idx": Value("int64"),
                    "html_attrs": {
                        "attrs": [Value("string")],
                        "values": [Value("string")],
                    },
                    "key": Value("string"),
                    "relative_end_pos": Value("int64"),
                    "relative_start_pos": Value("int64"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ],
            self.col_to_store_text: Value("string"),
            self.col_to_store_footer: [Value("string")],
            self.col_to_store_head: [Value("string")],
            self.col_to_store_title: [Value("string")],
        }
        return features

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        tags_to_remove_with_content = [
            html_parser.objects.TagToRemoveWithContent(tag="script"),
            html_parser.objects.TagToRemoveWithContent(tag="style"),
            html_parser.objects.TagToRemoveWithContent(tag="header"),
            html_parser.objects.TagToRemoveWithContent(tag="iframe"),
            html_parser.objects.TagToRemoveWithContent(
                tag="footer"
            ),  # copyright in footer
            html_parser.objects.TagToRemoveWithContent(tag="form"),
            html_parser.objects.TagToRemoveWithContent(
                tag="body", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="div", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="p", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="section", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="table", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="ul", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="ol", content_max_char_length=64
            ),
            html_parser.objects.TagToRemoveWithContent(
                tag="dl", content_max_char_length=64
            ),
        ]
        head_tag = "head"
        footer_tag = "footer"
        title_tag = "title"

        new_texts = []
        new_head = []
        new_footer = []
        new_title = []
        new_metadata = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_html]))]
        )
        for example_doc_html, example_metadata in zip(
            examples[self.col_html], new_metadata
        ):
            if example_doc_html is not None:
                (
                    plain_text,
                    metadata,
                    additional_columns,
                ) = html_parser.get_clean_text_and_metadata(
                    example_doc_html,
                    tags_to_remove_with_content=tags_to_remove_with_content,
                    consecutive_tags_to_fold=["div"],
                    convert_br_tag_to_breaking_line=True,
                    tags_sub_tree_to_isolate=[head_tag, footer_tag, title_tag],
                )
                new_texts.append(plain_text)
                new_head.append(additional_columns.get(head_tag, []))
                new_footer.append(additional_columns.get(footer_tag, []))
                new_title.append(additional_columns.get(title_tag, []))
                example_metadata.extend(
                    [
                        html_parser.objects.convert_html_metadata_dataclass_to_dict(
                            node
                        )
                        for node in metadata
                    ]
                )
            else:
                new_texts.append(None)
                new_head.append([])
                new_footer.append([])
                new_title.append([])

        examples[self.col_to_store_text] = new_texts
        examples[self.col_to_store_metadata] = new_metadata
        examples[self.col_to_store_head] = new_head
        examples[self.col_to_store_footer] = new_footer
        examples[self.col_to_store_title] = new_title
        return examples


class ErrorWrapperPreprocessor:
    def __init__(
        self,
        metadata_preprocessor: MetadataPreprocessor,
        output_keys: Dict[str, Any],
        verbose: bool = True,
    ) -> None:
        self.metadata_preprocessor = metadata_preprocessor
        self.output_keys = output_keys
        self.verbose = verbose

        self.error_column_name = f"{type(metadata_preprocessor).__name__}_error"
        self.error_comment_column_name = (
            f"{type(metadata_preprocessor).__name__}_error_comment"
        )

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        features = self.metadata_preprocessor.new_columns_minimal_features
        features.update(
            {
                self.error_column_name: Value("int64"),
                self.error_comment_column_name: Value("string"),
            }
        )
        return features

    def preprocess(self, examples: Dict[str, List]) -> Tuple[Dict[str, List], int]:
        """Process a batch of examples and add or extract corresponding metadata."""
        num_errors = 0

        metadata_list_backup = {
            col_name: copy.deepcopy(examples[col_name])
            for col_name in self.metadata_preprocessor.new_columns_minimal_features.keys()
            if col_name in examples
        }
        try:
            processed_examples = self.metadata_preprocessor.preprocess(
                examples=examples
            )

            random_key = list(processed_examples)[0]
            num_examples = len(processed_examples[random_key])
            if self.error_column_name not in processed_examples:
                processed_examples[self.error_column_name] = [
                    0 for _ in range(num_examples)
                ]

            if self.error_comment_column_name not in processed_examples:
                processed_examples[self.error_comment_column_name] = [
                    "" for _ in range(num_examples)
                ]
        except:  # noqa
            # we try the example one by one to find the culprit(s) and strore the error
            processed_examples = {
                key: []
                for key in list(self.output_keys.keys())
                + [self.error_column_name, self.error_comment_column_name]
            }

            for key, values in metadata_list_backup.items():
                examples[key] = copy.deepcopy(values)

            random_key = list(examples)[0]
            for idx in range(len(examples[random_key])):
                example = {key: [values[idx]] for key, values in examples.items()}
                try:
                    processed_example = self.metadata_preprocessor.preprocess(
                        examples=example
                    )

                    for key, value in processed_example.items():
                        processed_examples[key].append(value[0])

                    processed_examples[self.error_column_name].append(0)
                    processed_examples[self.error_comment_column_name].append("")
                except Exception as e:
                    for output_key in self.output_keys.keys():
                        if output_key in metadata_list_backup:
                            # We keep the initial value
                            processed_examples[output_key].append(
                                metadata_list_backup[output_key][idx]
                            )
                        elif output_key in example:
                            # We keep the initial value
                            processed_examples[output_key].append(
                                example[output_key][0]
                            )
                        else:
                            # We use the default value
                            processed_examples[output_key].append(
                                self.output_keys[output_key]
                            )

                    processed_examples[self.error_column_name].append(1)
                    processed_examples[self.error_comment_column_name].append(str(e))
                    logger.info(f"An error occurred with the message: {str(e)}")
                    num_errors += 1
        if self.verbose and num_errors != 0:
            logger.warning(f"{num_errors} errors occurred during the preprocessing")
        return processed_examples


@dataclass
class PreprocessingConfig:
    task_id: int = field(metadata={"help": "The id of the task"})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    num_files_to_process: int = field(
        metadata={"help": "the number of files to process"}
    )
    path_wiki_db: Optional[str] = field(
        metadata={
            "help": "The path to the wikipedia database file necessary for the website descriptions"
        }
    )
    entity_path_data_dir: Optional[str] = field(
        metadata={
            "help": "The path to the directory containing the directories `ed-wiki-2019`, `generic` and `wiki_2019` "
        }
    )
    path_or_url_flair_ner_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "TThe path or name of the flair ner model to use to preprocess entities"
        },
    )
    metadata_to_include: Optional[list] = field(
        default_factory=lambda: ["website_description", "entity", "timestamp"],
        metadata={"help": "The list of metadata to extract"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)"
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether the local cache containing datasets should be overwritten."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3?"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    map_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "This is the size of the batch size that will be used for the mapping operation when generating"
            " the dataset. If you are using `with_metadata` the recommended batch size is 1.."
        },
    )
    project_name: str = field(
        default="metadata_lm_exploration", metadata={"help": "The project name."}
    )
    save_batch_size: int = field(
        default=datasets.config.DEFAULT_MAX_BATCH_SIZE,
        metadata={"help": " Size of the batch to load in memory and write at once."},
    )
    use_load_from_disk: bool = field(
        default=False,
        metadata={
            "help": "If false, the program will load the dataset with `load_dataset` and if false, it will load it "
            "with `load_from_disk`."
        },
    )


class Logger:
    def __init__(self, *args, **kwargs):
        self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        wandb.log(dic)

    def close(self):
        wandb.finish()


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


col_html = "html_str"
col_url = "url"
col_to_store_text = "text"
col_to_store_head = "html_head"
col_to_store_footer = "html_footer"
col_to_store_title = "html_title"
col_to_store_metadata_html = "metadata_html"


@hydra.main(config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:  # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config_dict = OmegaConf.to_container(args)
    metrics_logger = Logger(project=args.project_name, config=config_dict)

    logger.info("Initialize the preprocessors:")
    if "html" in args.metadata_to_include:
        logger.info("   Html...")
        _html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html,
            col_to_store_text=col_to_store_text,
            col_html=col_html,
            col_to_store_footer=col_to_store_footer,
            col_to_store_head=col_to_store_head,
            col_to_store_title=col_to_store_title,
        )
        html_processor = ErrorWrapperPreprocessor(
            metadata_preprocessor=_html_processor,
            output_keys={
                col_to_store_metadata_html: [],
                col_to_store_text: "",
                col_to_store_footer: [],
                col_to_store_head: [],
                col_to_store_title: [],
            },
        )

    logger.info("Processors initialization finished")

    poss_files = sorted(os.listdir(args.dataset_name))

    if args.use_load_from_disk:
        poss_files = [
            file_name
            for file_name in poss_files
            if file_name.startswith("pseudo_crawl_seed")
        ]
    else:
        poss_files = [
            file_name
            for file_name in poss_files
            if (file_name.endswith("jsonl.gz") or file_name.endswith("jsonl"))
            and file_name.startswith("pseudo_crawl_seed")
        ]

    def process_file(file_name: str):

        logger.info(config.HF_DATASETS_CACHE)
        processing_name = (
            "-".join(args.metadata_to_include)
            if args.metadata_to_include is not None
            else "full-process"
        )
        metrics_logger.log({processing_name: 0})

        metrics_logger.log({"load_dataset": 0})
        if args.use_load_from_disk:
            dataset_name = os.path.join(args.dataset_name, file_name)
            logger.info(f"Loading the dataset {dataset_name} with `load_from_disk`")
            ds = load_from_disk(dataset_name)
        else:
            data_files = {"file": file_name}
            logger.info(
                "Loading a dataset with `load_dataset`"
                f"{args.dataset_name}, {args.dataset_config_name}, data_files={data_files}, cache_dir={args.cache_dir},"
            )
            ds = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                data_files=data_files,
                cache_dir=args.cache_dir,
                keep_in_memory=False,
                download_mode="force_redownload",
            )["file"]

        metrics_logger.log({"load_dataset": 1})

        features_dict = dict(ds.features)
        logger.info(f"the initial features of the dataset are: {features_dict}")

        def apply_processor(
            ds: Dataset, processor: MetadataPreprocessor, remove_columns=None
        ) -> Dataset:
            for (
                col_name,
                feature_type,
            ) in processor.new_columns_minimal_features.items():
                assert col_name not in features_dict
                features_dict[col_name] = feature_type
            extraction_name = processor.__class__.__name__

            logger.info(f"Start {extraction_name}")
            metrics_logger.log({extraction_name: 0})
            ds = ds.map(
                processor.preprocess,
                batched=True,
                batch_size=args.map_batch_size,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Running {extraction_name} on dataset",
                features=Features(features_dict),
                remove_columns=remove_columns,
            )
            metrics_logger.log({extraction_name: 1})
            logger.info(f"End {extraction_name}")
            return ds

        if "html" in args.metadata_to_include:
            ds = apply_processor(ds=ds, processor=html_processor)

        if file_name.endswith(".jsonl.gz"):
            out_file_name = file_name[: -len(".jsonl.gz")]
        elif file_name.endswith(".jsonl"):
            out_file_name = file_name[: -len(".jsonl")]
        else:
            out_file_name = file_name
        out_file_name_tmp = f"tmp-{out_file_name}"

        saving_path = os.path.join(args.out_dir, out_file_name)
        saving_path_tmp = os.path.join(args.out_dir, out_file_name_tmp)

        logger.info(f"Save resulting dataset {ds} at {saving_path_tmp}")
        metrics_logger.log({"save_result": 0})
        ds.save_to_disk(saving_path_tmp)
        metrics_logger.log({"save_result": 1})
        logger.info(f"Moving the saved dataset to {saving_path}")
        subprocess.run(["mv", saving_path_tmp, saving_path])
        logger.info(f"Processing of {file_name} ended successfully")
        metrics_logger.log({processing_name: 1})

    for file_name in poss_files[
        args.task_id
        * args.num_files_to_process : args.task_id
        * args.num_files_to_process
        + args.num_files_to_process
    ]:
        logger.info(f"Start to process {file_name}")
        process_file(file_name=file_name)

    metrics_logger.close()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
