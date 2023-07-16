from llmeval.infer import register_llm_infer
@register_llm_infer("hello")
def say_hello(name_ls,
              ptm_name=None,
              peft_name=None):
    print(f"ptm_name:{ptm_name}")
    print(f"peft_name:{peft_name}")
    res = []
    for name in name_ls:
        res.append(f"hello: {name}")
    return res


@register_llm_infer("repeat")
def repeat(name_ls,
          ptm_name=None,
          peft_name=None):
    print(f"ptm_name:{ptm_name}")
    print(f"peft_name:{peft_name}")
    res = []
    for name in name_ls:
        res.append(name)
    return res


