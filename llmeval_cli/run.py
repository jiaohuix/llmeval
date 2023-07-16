'''
input: text
output: json,excel
0.input:  question.txt
1.chatgpt baseline: question.txt -> answer_gpt35.jsonl
2.llm infer: answer_${infer_name}.jsonl
3.gpt4 review: ()-> review_${infer_name}.jsonl
4.merge different model to excel
'''
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import logging
from tqdm import tqdm
from llmeval.file_io import read_file, merge_excels, write_alpaca_eval, read_jsonl
from llmeval.options import get_parser
from llmeval.baseline import baseline_inference
from llmeval.infer import llm_inference
from llmeval.review import review_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_infer_names(args):
    from llmeval.infer import INFER_REGISTRY
    all_regis = ",".join(list(INFER_REGISTRY.keys()))
    for infer_name in args.infer_names:
        assert infer_name in INFER_REGISTRY.keys(), f"{infer_name} not in INFER_REGISTRY: [{all_regis}]."

def main():
    # parse args
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    check_infer_names(args)

    alpaca_eval_res = []
    alpaca_eval_chatgpt_res = []

    # read data
    logger.info(f"reading data...")
    questions = read_file(args.input)

    # chatgpt baseline
    gpt35_outfile = os.path.join(args.outdir,"answer_gpt35.jsonl")
    logger.info(f"Stage1: Infer chatgpt answer ing...")
    if not os.path.exists(gpt35_outfile):
        answer_gpt35 = baseline_inference(args)
    else:
        answer_gpt35 = read_jsonl(gpt35_outfile)
    if  args.alpaca_eval:
        for ques, ans in zip(questions, answer_gpt35):
            data = {
                "instruction": ques,
                "input": "",
                "output": ans["text"] if isinstance(ans,dict) else ans,
                "generator": "gpt-3.5-turbo",
                "dataset": "helpful_base",
                "datasplit": "eval"
            }
            alpaca_eval_chatgpt_res.append(data)
        gpt35_outfile = os.path.join(args.outdir, f"output_gpt35.json")
        write_alpaca_eval(alpaca_eval_chatgpt_res, file=gpt35_outfile)

    # infer llm and gpt4 review
    logger.info(f"Stage2: Infer&Review llm answer ing...")
    for idx,infer_name in tqdm(enumerate(args.infer_names)):
        logger.info(f"Stage2.{idx+1}: Infer {infer_name} ing...")
        llm_outfile = os.path.join(args.outdir,f"answer_{infer_name}.jsonl")
        if not os.path.exists(llm_outfile):
            answer_llm = llm_inference(text_ls=questions,infer_name=infer_name,
                                       yamlfile=args.conf, outfile=llm_outfile)
        else:
            answer_llm = read_jsonl(llm_outfile)
        if args.alpaca_eval:
            for ques, ans in zip(questions, answer_llm):
                data = {
                    "instruction": ques,
                    "input": "",
                    "output": ans["text"] if isinstance(ans,dict) else ans,
                    "generator": infer_name,
                    "dataset": "helpful_base",
                    "datasplit": "eval"
                }

                alpaca_eval_res.append(data)

        # gpt4 review
        if not args.alpaca_eval:
            logger.info(f"Stage3.{idx+1}: Review {infer_name} ing...")
            review_outfile = os.path.join(args.outdir, f"review_{infer_name}.jsonl")
            if not os.path.exists(review_outfile):
                review = review_inference(args, infer_name=infer_name)

    # merge excel
    if not args.alpaca_eval:
        logger.info(f"Stage4: Merge excel result ing...")
        merge_excels(outdir=args.outdir, outfile="merged_llmeval.xlsx")
    else:
        outfile = os.path.join(args.outdir, f"output.json")
        write_alpaca_eval(alpaca_eval_res, file=outfile)


if __name__ == '__main__':
    main()

