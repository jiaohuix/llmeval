'''
功能：负责参数的配置。
参数：模型、问题列表、输出地址
输出：json及excel
使用yaml配置参数，然后用hydra加载参数并可以命令行覆盖

1 基础参数： 输入文件、输出文件、（可选：是否合并excel、是否加载已经存在的参考答案）
2 llm_infer参数： llm_name, 权重路径llm_ckpt（可选）
3 chatgpt参数: model_name=gpt35, max_tokens,MAX_API_RETRY = 5  REQ_TIME_GAP = 10
4 gpt4 review参数(同chatgpt)，额外加上是否翻译中文tozh

'''
import argparse

def add_openai_args(parser):
    group = parser.add_argument_group("Openai")
    # fmt: off
    parser.add_argument("--tozh",
                        action="store_true",
                        help = "Weather translate result to chinese.")
    group.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature",
    )

    group.add_argument(
        "--max-api-retry",
        type=int,
        default=5,
        help="maximum api retry",
    )

    group.add_argument(
        "--req-time-gap",
        type=int,
        default=10,
        help="REQ_TIME_GAP",
    )
    # fmt: on
    return parser

def add_llm_args(parser):
    group = parser.add_argument_group("LLM")
    # fmt: off
    group.add_argument("--llm-ckpt", default=None,
                       help="llm ckpt")
    group.add_argument("--peft-ckpt", default=None,
                       help="peft ckpt")
    # fmt: on
    return parser


def get_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # io
    parser.add_argument("-i", "--input", default=None,required=True,
                       help="input question text file")
    parser.add_argument("-o", "--outdir", default=None,required=True,
                       help="output directory") # 含：answer_gpt35.json, answer_${llm_name}

    parser.add_argument("-c", "--conf", default=None,
                       help="yaml file path.")

    parser.add_argument("-a", "--alpaca-eval", action="store_true",
                       help="yaml file path.")
    # llm infer
    from llmeval.infer import INFER_REGISTRY
    parser.add_argument(
        "--infer-names", required=True,
        nargs="+", default=[],
        # choices=INFER_REGISTRY.keys(),
    )

    # chatgpt
    parser.add_argument("--base-model", default="gpt-3.5-turbo",
                       help="baseline model name, default=chatgpt(gpt35)")
    # gpt4
    # parser.add_argument("--review-model", default="gpt-4",
    parser.add_argument("--review-model", default="gpt-3.5-turbo",
                       help="baseline model name, default=chatgpt(gpt35)")
    # add args
    parser = add_openai_args(parser)
    parser = add_llm_args(parser)
    return parser

