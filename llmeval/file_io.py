'''
功能：负责文件的读写。
'''
import os
import json
import pandas as pd


def read_file(file):
    # 读取文本文件
    with open(file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip() != ""]
    return lines


def write_file(res, file):
    # 写文本文件
    base_dir = os.path.dirname(file)
    if base_dir == "":
        file = "./" + file
    elif not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(res)
    print(f'write to {file} success, total {len(res)} lines.')


# jsonl: 文本文件，每行是json字符串
def write_jsonl(res, file):
    js_res = [json.dumps(r, ensure_ascii=False) + "\n" for r in res]
    write_file(js_res, file)


def read_jsonl(file):
    lines = read_file(file)
    data = []
    for line in lines:
        line = line.strip()
        if not line: continue
        js_data = json.loads(line)
        data.append(js_data)
    return data


def write_alpaca_eval(data=[],file="output.json"):
    '''alpaca_eval format'''
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'write to {file} success, total {len(data)} lines.')

def load_questions(file):
    # text -> json_list= {"question_id": 0,
    #                     "text":"aabbcc"}
    json_list = []
    for idx, line in enumerate(read_file(file)):
        line = {
            "question_id": idx,
            "text": line
        }
        json_list.append(line)
    return json_list


def write_excel(js_data, outfile="res.xlsx"):
    # js_data: 包含字典的列表，字典字段必须相同
    df = pd.DataFrame.from_dict(js_data)
    df.to_excel(outfile, index=False)
    print(f'write to {outfile} success, total {len(js_data)} lines.')


def merge_excels(outdir, outfile="merged.xlsx"):
    out_path = os.path.join(outdir, outfile)

    parse_infer_name_fn = lambda name: name.replace("review_", "").replace(".xlsx", "")

    # 1.find all review result
    file_names = [name for name in os.listdir(outdir)
                  if name.endswith(".xlsx")
                  and name.find("review_") != -1]
    assert len(file_names) >= 1, "review excel result num must greater than 0."
    # file_paths = [os.path.join(outdir,name) for name in file_names]

    # 2. read excel
    df_dict = {}
    last_len = -1
    for name in file_names:
        file_path = os.path.join(outdir, name)
        infer_name = parse_infer_name_fn(name)
        df = pd.read_excel(file_path)
        df_dict[infer_name] = df
        # df_dict[name] = df

        # check length
        cur_len = len(df)
        if last_len != -1:
            assert cur_len == last_len, "review excel lines not equal."
            last_len = cur_len

    # 3. merge excel
    # field: prompt, qid, model(ans, explain), score_idx,
    df1 = df_dict[list(df_dict.keys())[0]]
    res_df = pd.DataFrame({"qid": df1["question_id"], "prompt": df1["question"]})
    for idx, (infer_name, df) in enumerate(df_dict.items(), start=1):
        # answer and comment
        res_df["answer"] = df["llm_answer"]
        res_df["comment"] = df["llm_explain"]
        res_df[infer_name] = res_df.apply(lambda x: 'Answer: {answer} \nComment: {comment}.'.format(
            answer=x['answer'], comment=x['comment']), axis=1)
        # score
        res_df[f"score{idx}"] = df["llm_score"]

    res_df.drop(labels=["answer", "comment"], inplace=True, axis=1)
    res_df.to_excel(out_path, index=False)
