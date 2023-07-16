from llmeval.infer import register_llm_infer


@register_llm_infer("baichuan_7b")
def infer_baichuan_7b(text_ls,
              ptm_name="baichuan-inc/Baichuan-7B",
              peft_name=None):
    print(f"ptm_name:{ptm_name}")
    print(f"peft_name:{peft_name}")
    res = []
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ptm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ptm_name, device_map="auto", trust_remote_code=True)
    for text in text_ls:
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to('cuda:0')
        pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        res.append(ans)
    return res




