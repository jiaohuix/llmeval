from llmeval.infer import register_llm_infer

@register_llm_infer("chatglm_6b")
def infer_chatglm_6b(text_ls,
              ptm_name="THUDM/chatglm-6b",
              peft_name=None):
    '''chatglm-6b的推理函数。

    Args:
        text_ls:list[str] 包含若干文本问题的列表。
        ptm_name:str   模型的标识，用来加载模型和权重。
        peft_name:str   lora或额外参数的加载路径，default=None。

    Returns:list[str]   包含若干答案的列表，由模型推理得到。

    '''
    from transformers import AutoTokenizer, AutoModel
    print(f"ptm_name:{ptm_name}")
    print(f"peft_name:{peft_name}")
    res = []
    tokenizer = AutoTokenizer.from_pretrained(ptm_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(ptm_name, trust_remote_code=True).half().cuda()
    for text in text_ls:
        response, history = model.chat(tokenizer, text, history=[])
        res.append(response)
    return res




