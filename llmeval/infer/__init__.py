'''
llm推理模块
'''
import os
import logging
import importlib
from functools import partial
from yacs.config import CfgNode
from llmeval.file_io import write_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INFER_REGISTRY = {}

def register_llm_infer(infer_name):
    """
    New llm inference function can be added to llmeval with the
    :func:`register_llm_infer` function decorator.

    For example::

        @register_llm_infer("hello")
        def llm_demo(text_ls,
                ptm_name=None,
                peft_name=None):
            res = []
            for text in text_ls:
                res.append(text)
            return res

    Args:
        infer_name (str): the name of the llm inference function.
    """
    def register_llm_infer_fn(fn):
        if infer_name in INFER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate llm inference function ({})".format(infer_name)
            )
        if not callable(fn):
            raise ValueError(
                "Inference function must be callable ({})".format(infer_name)
            )
        INFER_REGISTRY[infer_name] = fn

        return fn

    return register_llm_infer_fn


def import_models(infer_dir, namespace):
    for file in os.listdir(infer_dir):
        path = os.path.join(infer_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            infer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + infer_name)


def llm_inference(text_ls, infer_name,yamlfile=None, outfile="res.jsonl"):
    all_regis = ",".join(list(INFER_REGISTRY.keys()))
    assert infer_name in INFER_REGISTRY.keys(), f"{infer_name} not in INFER_REGISTRY: [{all_regis}]."
    # 1 load yaml config
    conf = {}
    if (yamlfile is not None) and os.path.exists(str(yamlfile)):
        conf = CfgNode.load_cfg(open(str(yamlfile), encoding="utf-8"))
    update_keys = list(conf.keys())
    # 2 get infer function
    if infer_name in update_keys:
        ptm_name = conf[infer_name].ptm
        peft_name = conf[infer_name].peft
        if peft_name == "None":
            peft_name = None


        logger.info(f" Update {infer_name}'s ptm_name:[{ptm_name}] and peft_name:[{str(peft_name)}] from yamlfile: [{yamlfile}] ")
        infer_fn = partial(INFER_REGISTRY[infer_name],
                           ptm_name = ptm_name,
                           peft_name = peft_name )
    else:
        infer_fn = INFER_REGISTRY[infer_name]
    # infer_fn = INFER_REGISTRY[infer_name]

    # 3 infer question
    res  = infer_fn(text_ls)

    # 4 save result
    js_res = []
    for qid,text in enumerate(res):
        tmp = {"question_id": qid, "text": text}
        js_res.append(tmp)
    write_jsonl(js_res,outfile)
    return res


# automatically import any Python files in the infer/ directory
infer_dir = os.path.dirname(__file__)
import_models(infer_dir, "llmeval.infer")


