import json
import requests
from llmeval.infer import register_llm_infer

@register_llm_infer("chatglm_6b_api")
def infer_chatglm_6b_api(text_ls,
          ptm_name="http://100.126.237.58:8000",
          peft_name=None):
    res = []
    for text in text_ls:
        data = {"prompt": text, "history": []}
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url=ptm_name, data=json.dumps(data), headers=headers)
        try:
            ret = json.loads(resp.content)["response"]
            print(ret)
        except:
            ret = "#ERROR#"
            print(ret)

        res.append(ret)
    return res