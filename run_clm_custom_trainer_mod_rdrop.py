import logging
import math
import os
import sys
from statistics import mean
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import torch
import torch.nn.functional as F

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    reinforce_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "control the weight of Cross-entropy and REINFORCE"},
    )

    r_drop_alpha: Optional[float] = field(
        default=0.0,
        metadata={"help": "control the weight of KL-divergence"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=model_args.cache_dir
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         extension,
        #         data_files=data_files,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         cache_dir=model_args.cache_dir,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         extension,
        #         data_files=data_files,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         cache_dir=model_args.cache_dir,
        #     )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Set pad token for gpt2
    model.resize_token_embeddings(len(tokenizer))
    print("len(tokenizer)", len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(
                examples[text_column_name],
                add_special_tokens=False,
                truncation=True,
                # padding="longest",
                max_length=data_args.block_size,
            )
            # output["labels"] = output["input_ids"]
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            batch_size=training_args.per_device_train_batch_size,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        # tokenized_datasets.remove_columns_(["token_type_ids"])
        # tokenized_datasets.set_format(type="torch")

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             "Picking 1024 instead. You can change that default value by passing --block_size xxx."
    #         )
    #         block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    block_size = data_args.block_size

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with training_args.main_process_first(desc="grouping texts together"):
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # 以batch為單位進行生成，batch內的長度要一樣，padding要pad在前方，例如： [PAD][PAD][CLS] dialogue_history 開始生成belief state，把[PAD]放在dialogue_history後方進行生成是錯誤的。
    left_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left"
    )
    MAX_LEN = model.config.n_ctx  # GPT-2 max length: 1024
    BS_MAX_LEN = 163
    model.config.eos_token_id = 29  # 設定模型只生成到 <|endofbelief|> 的 ">" ID:29

    class RTCTODTrainer(Trainer):
        def calc_acc_hard(self, pred, target):
            if pred == target:
                return 1.0
            return 0.0

        def clean_bs(self, belief):
            clean_tokens = ["<|endoftext|>", "", " "]
            new_belief = []###
            print("===ord_belief===", belief)
            for bs in belief:
                if bs in clean_tokens or "none" in bs or "mentioned" in bs:
                    continue
                new_belief.append(bs.strip())###
            print("===new_belief===", list(sorted(set(new_belief))))
            return list(sorted(set(new_belief)))

        def find_eos_index(self, beam_output):
            index = -1
            for i in range(len(beam_output))[::-1]:
                if beam_output[i - 1] == 29:
                    index -= 1
                else:
                    return index

        def greedy(self, batch):
            input_ids = batch["input_ids"]
            input_text_batch = tokenizer.batch_decode(
                input_ids, skip_special_tokens=False
            )
            del input_ids
            torch.cuda.empty_cache()

            target_belief_states = []
            for input_text in input_text_batch:
                target_belief_state = input_text.split("<|belief|>")[-1].split(
                    "<|endofbelief|>"
                )[0]
                target_belief_states.append(target_belief_state)
            target_belief_states = [bs.split(",") for bs in target_belief_states]

            target_belief_states = [
                [b.strip() for b in bs] for bs in target_belief_states
            ]

            history_text_batch = [
                input_text.split("<|belief|>")[0] + "<|belief|>"
                for input_text in input_text_batch
            ]
            history_tensor_batch = left_tokenizer(
                history_text_batch,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

            # for indexed_tokens in history_tensor_batch:
            #     if len(indexed_tokens) > MAX_LEN:
            #         indexed_tokens = indexed_tokens[-1 * MAX_LEN :]

            history_tensor_batch = history_tensor_batch.to(model.device)
            print("org_history_tensor_batch_shape:", history_tensor_batch.shape) #####8/16增加
            gen_max_len = MAX_LEN
            batch_seq_len = history_tensor_batch.shape[1]
            if batch_seq_len + BS_MAX_LEN < MAX_LEN:
                gen_max_len = batch_seq_len + BS_MAX_LEN
                if gen_max_len == 1024:
                    history_tensor_batch = history_tensor_batch[:, :-1]   #####8/16增加
            print("gen_max_len:", gen_max_len)#####8/16增加
            print("after_history_tensor_batch_shape:", history_tensor_batch.shape)
            ##################### beam search後，把belief state抓出來 #####################
            # 之後要用來計算 JGA 的
            num_return_sequences = 3
            beam_outputs = model.generate(
                history_tensor_batch,
                num_beams=num_return_sequences,
                num_return_sequences=num_return_sequences,
                max_length=gen_max_len,
                early_stopping=True,
                pad_token_id=29,
                return_dict_in_generate=True,
                output_scores=True,
            )

            del history_tensor_batch
            torch.cuda.empty_cache()

            greedy_belief_states = []
            for i in range(0, len(beam_outputs.sequences), num_return_sequences):
                tmp = []
                id_outputs = beam_outputs.sequences[i : i + num_return_sequences]
                for id in id_outputs:
                    history_belief = tokenizer.decode(id, skip_special_tokens=False)
                    belief = (
                        history_belief.split(" <|endofcontext|>")[-1]
                        .split("<|belief|>")[-1]
                        .split("<|endofbelief|>")[0]
                    )
                    tmp.append(belief)
                greedy_belief_states.append(tmp)
            #############################################################################

            ##################### beam search後，把logits處理一下 #########################
            """
            由於huggingface支援batch generation，因此生成出來的每個句子長度都會一樣。
            但其中有些句子可能belief state早就生成出來了，因此huggingface會利用你上面
            model.generate 設定的 pad_token_id=29，來去作padding，以確保所有句子最後長度都一樣。
            pad_token_id=29，在英文GPT-2的字典裡面代表 ">" 這個符號，這邊的想法是
            當模型生成到 <|endofbelief|> 最後的 ">" 就算結束，因此model.generate會把提早生成出belief state的語句
            利用 ">" 進行padding，因此某些句子會呈現 dialogue_history -> belief_state -> ">>>>>>>>>>" 生成一堆 ">" 來補齊長度。

            如果之後有自訂義的token id要停下來，請把程式裡面所有出現29的地方進行修改。如521,541,597行。

            因此，以下的程式我們要將每個生成句多餘的 ">" 清除。
            """
            batch_logits = torch.stack(beam_outputs.scores, dim=1).to("cpu")
            # for logits in batch_logits:
            #     print(logits.shape)  # [number_of_generated_token, 50258 字典大小]

            # 抓出每個句子真正的end index，也就是抓到 <|endofbelief|> 的 ">" 才是真正的 end token
            batch_eos_index = [
                self.find_eos_index(beam_output)
                for beam_output in beam_outputs.sequences
            ]

            truncate_logits = []
            truncate_actions = []
            for actions, logits, index in zip(
                beam_outputs.sequences, batch_logits, batch_eos_index
            ):
                if index != -1:
                    truncate_logits.append(logits[: index + 1])
                    truncate_actions.append(actions[-len(logits) : index + 1].to("cpu"))
                else:
                    truncate_logits.append(logits)
                    truncate_actions.append(actions[-len(logits) :].to("cpu"))

            batch_logits = [
                truncate_logits[i : i + num_return_sequences]
                for i in range(len(truncate_logits))[::num_return_sequences]
            ]
            batch_actions = [
                truncate_actions[i : i + num_return_sequences]
                for i in range(len(truncate_actions))[::num_return_sequences]
            ]

            return (
                target_belief_states,
                greedy_belief_states,
                batch_logits,
                batch_actions,
            )

        def reinforce(
            self,
            target_belief_states,
            greedy_belief_states,
            batch_logits,
            batch_actions,
        ):
            net_policies = []
            net_actions = []
            net_advantages = []
            avg_reward_baseline = []
            batch_accs = []

            for turn_pred, turn_target in zip(
                greedy_belief_states, target_belief_states
            ):
                turn_target = self.clean_bs(turn_target)
                for t_p in turn_pred:
                    t_p = t_p.split(",")
                    self.clean_bs(t_p)

                clean_turn_pred = [self.clean_bs(t_p.split(",")) for t_p in turn_pred]
                accs = [
                    self.calc_acc_hard(pred, turn_target) for pred in clean_turn_pred
                ]
                batch_accs.append(accs)

                # 對應PPT論文38頁 Q(s) 0.33那欄
                reward_baseline = mean(accs)

                # 假如有3個Dialogue，for loop跑完後 avg_reward_baseline 就會有 3 個平均後的 baseline
                avg_reward_baseline.append(reward_baseline)

            for avg, accs, logits, actions in zip(
                avg_reward_baseline,
                batch_accs,
                batch_logits,
                batch_actions,
            ):
                """
                這裡的for loop 會跑完所有的dialogue，也就是你的batch size設多少，這裡的for就會跑幾次。
                """
                # 這裡的 actions 是單一個 dialogue 用beam search生成的多個結果，以口試講義來說，這裡就是包含三組結果。
                # print("actions", actions)

                # 這裡的 logits 是上面每個 actions 對應的 logits
                # print("logits", logits)

                # 每個 beam search 的 JGA 結果，1分或0分，這裡代表 每個人的Q(s)
                # print("accs", accs)

                # 這裡的avg代表，這一個 turn 的多個 Q(s) 平均值: "b"
                # print("avg", avg)

                for acc, l, a in zip(accs, logits, actions):
                    """
                    這裡的for loop 是針對單一個dialogue_history進行beam search的結果。
                    如果beam size是3，這裡會跑三次迴圈。
                    """
                    net_policies.append(l)

                    net_actions.extend(a)

                    adv = acc - avg  # 對應論文PPT 38頁 Q(s) - b

                    net_advantages.extend([adv] * len(a))
                    # 每個 Token 都要乘上各自的 belief_state 對應的 Q(s) - b
                    # 譬如說第一個belief state 生成出 餐廳價格範圍便宜的，這 9 個 token(action) 的 logp(a|s) 都要乘上同一個 Q(s) - b

            ##### Policy Loss Calculation #####
            # Calculate the policy gradient by applying log_softmax(使用 log_softmax 的好處是，運算梯度時數值會更穩定)
            # and choose values from the chosen actions, scaled by their advantages.
            # The negative mean of those scaled logarithms will be our loss that we ask the optimizer to minimize.

            # 將所有batch_action 的 logits 沿著第 0 維串接
            policies_v = torch.cat(net_policies)
            # policies_v.shape [total number of actions, vocab_size]
            # print("policies_v", policies_v.shape)

            # 對每個 action 的 logits 取softmax+log: log_p(a|s)
            log_prob_v = F.log_softmax(policies_v, dim=-1)

            del policies_v
            torch.cuda.empty_cache()

            actions_t = torch.LongTensor(net_actions).to(model.device)
            # print("actions_t", actions_t.shape) # actions_t.shape [all_samples_action]

            adv_v = torch.FloatTensor(net_advantages).to(model.device)  # [Q(s) - b]

            # log p(a|s)，只取出該 action 的 log_prob
            lp_a = log_prob_v[range(len(net_actions)), actions_t].to(model.device)

            log_prob_actions_v = adv_v * lp_a  # [Q(s) - b] * log_p(a|s)

            del actions_t
            torch.cuda.empty_cache()
            del log_prob_v
            torch.cuda.empty_cache()
            del adv_v
            torch.cuda.empty_cache()
            del lp_a
            torch.cuda.empty_cache()

            rl_loss = -log_prob_actions_v.mean()

            del log_prob_actions_v
            torch.cuda.empty_cache()

            return rl_loss

        def find_pad_index(self, input_mask):
            print("beam_output:", input_mask)
            index = -1
            for i in range(len(input_mask))[::-1]:
                if input_mask[i-1] == 50257:
                    index -= 1
                elif index == -1:
                    index = 0
                else:
                    return index

        def compute_kl_loss(self, p, q, batch_pad_index=None):
            p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
            q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

            del p
            torch.cuda.empty_cache()
            del q
            torch.cuda.empty_cache()
            # print("p_loss", p_loss)
            print("p_loss")
            for i in p_loss:
                print(i)
            print("==="*50)
            print("q_loss")
            # for i in q_loss:
            #     print(i)
            # print("q_loss", q_loss)
            print("p_loss.size", p_loss.size())
            print("q_loss.size", q_loss.size())

            if batch_pad_index is not None:
                for i in range(len(batch_pad_index)):
                    j = batch_pad_index[i]
                    print("new_jjjj", j)
                    if j != 0:
                        p_loss[i][j:][:] = 0.
                        q_loss[i][j:][:] = 0.
            print("new_p_loss")
            for i in p_loss:
                print(i)
            print("new_p_loss.size", p_loss.size())
            p_loss = p_loss.sum(-1).mean()
            print("sum_p_loss", p_loss)
            print("***" * 50)
            print("new_q_loss")
            # for i in q_loss:
            #     print(i)
            print("new_q_loss.size", q_loss.size())
            q_loss = q_loss.sum(-1).mean()
            print("sum_q_loss", q_loss)
            loss = (p_loss + q_loss) / 2
            print("pq_sum_loss:", loss)
            del p_loss
            torch.cuda.empty_cache()
            del q_loss
            torch.cuda.empty_cache()

            return loss

        def compute_loss(self, model, inputs, return_outputs=False):
            # If apply REINFORCE algorithm
            alpha = model_args.reinforce_alpha
            r_drop_alpha = model_args.r_drop_alpha
            if 1.0 - alpha:
                (
                    target_belief_states,
                    greedy_belief_states,
                    batch_logits,
                    batch_actions,
                ) = self.greedy(inputs)

                rl_loss = self.reinforce(
                    target_belief_states,
                    greedy_belief_states,
                    batch_logits,
                    batch_actions,
                )

                inputs["labels"] = torch.where(
                    inputs["input_ids"] == tokenizer.pad_token_id,
                    -100,
                    inputs["input_ids"],
                )
                outputs = model(**inputs)
                ce_loss = outputs.loss

                del inputs
                torch.cuda.empty_cache()
                del outputs
                torch.cuda.empty_cache()

                total_loss = alpha * ce_loss + (1 - alpha) * rl_loss
                return (total_loss, outputs) if return_outputs else total_loss

            elif r_drop_alpha:
                print("R-Drop***********************")
                batch_pad_index = [
                    self.find_pad_index(input_mask)
                    for input_mask in inputs["input_ids"]
                ]
                print("batch_pad_index:", batch_pad_index)
                # *****
                # for i in range(len(batch_pad_index)):
                #     # for j in batch_pad_index[i]:
                #         j = batch_pad_index[i]
                #         # print("jjjj",j)
                #         # if inputs["input_ids"][i][j] == 0:
                #         if j != 0:
                #             inputs["input_ids"][i][j:] = 100
                # *****
                # newwwww = [self.pad_2_none(batch_pad_index, i) for i in inputs["input_ids"]]
                # print("newwwww" ,inputs["input_ids"])
                # /////
                # pad_mask = torch.where(inputs["input_ids"] == tokenizer.pad_token_id, 1, 0)
                # print("pad_mask", pad_mask)
                # print("pad_mask.size()", pad_mask.size())
                # /////

                inputs["labels"] = torch.where(
                    inputs["input_ids"] == tokenizer.pad_token_id,
                    -100,
                    inputs["input_ids"],
                )
                logits = model(**inputs)
                logits2 = model(**inputs)

                ce_loss = 0.5 * (logits.loss + logits2.loss)
                print("ce_loss", ce_loss)

                # print("logits.logits", logits.logits)
                # print("logits2.logits", logits2.logits)
                kl_loss = self.compute_kl_loss(logits.logits, logits2.logits, batch_pad_index=batch_pad_index)

                del inputs
                torch.cuda.empty_cache()
                del logits
                torch.cuda.empty_cache()
                del logits2
                torch.cuda.empty_cache()
                print("kl_loss", kl_loss)
                total_loss = ce_loss + r_drop_alpha * kl_loss
                print("total_loss", total_loss)
                return total_loss

            else:
                inputs["labels"] = torch.where(
                    inputs["input_ids"] == tokenizer.pad_token_id,
                    -100,
                    inputs["input_ids"],
                )
                outputs = model(**inputs)
                ce_loss = outputs.loss

                return (ce_loss, outputs) if return_outputs else ce_loss

    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    # Initialize our Trainer
    trainer = RTCTODTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "text-generation",
        }
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs[
                    "dataset"
                ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()