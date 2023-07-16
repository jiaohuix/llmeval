# # -*- coding: utf-8 -*-
# """
# @author:XuMing(xuming624@qq.com)
# @description:
# """
#
# import os
# import sys
# from llmeval.infer import register_llm_infer
# import pandas as pd
# from textgen import LlamaModel, ChatGlmModel
#
#
# def llama_generate_prompt(instruction):
#     return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:{instruction}\n\n### Response: """
#
# def chatglm_generate_prompt(instruction):
#     return f"""{instruction}\n答："""
#
# @register_llm_infer(infer_name="llama_13b_lora")
# def infer_llama_13b_lora(text_ls,
#                          ptm_name="decapoda-research/llama-13b-hf",
#                          peft_name='shibing624/llama-13b-belle-zh-lora'):
#     m = LlamaModel('llama',
#                    model_name = ptm_name,
#                    peft_name = peft_name)
#
#     predict_sentences = [llama_generate_prompt(s) for s in text_ls]
#     res = m.predict(predict_sentences)
#     return res
#
# @register_llm_infer(infer_name="llama_7b_alpaca_plus")
# def infer_llama_7b_alpaca_plus(text_ls,
#                          ptm_name="shibing624/chinese-alpaca-plus-7b-hf",
#                          peft_name=None):
#     m = LlamaModel('llama',
#                    model_name = ptm_name,
#                    peft_name = peft_name,
#                    args={'use_peft': False})
#     predict_sentences = [llama_generate_prompt(s) for s in text_ls]
#     res = m.predict(predict_sentences)
#     return res
#
# @register_llm_infer(infer_name="llama_13b_alpaca_plus")
# def infer_llama_13b_alpaca_plus(text_ls,
#                              ptm_name="shibing624/chinese-alpaca-plus-13b-hf",
#                              peft_name=None):
#     m = LlamaModel('llama',
#                    model_name = ptm_name,
#                    peft_name = peft_name,
#                    args={'use_peft': False})
#     predict_sentences = [llama_generate_prompt(s) for s in text_ls]
#     res = m.predict(predict_sentences)
#     return res
#
# @register_llm_infer(infer_name="chatglm_6b")
# def infer_chatglm_6b(text_ls,
#                      ptm_name="THUDM/chatglm-6b",
#                      peft_name=None):
#     m = ChatGlmModel('chatglm',
#                    model_name = ptm_name,
#                    peft_name = peft_name,
#                    args={'use_peft': False})
#     predict_sentences = [chatglm_generate_prompt(s) for s in text_ls]
#     res = m.predict(predict_sentences)
#     return res
#
# @register_llm_infer(infer_name="chatglm_6b_lora")
# def infer_chatglm_6b_lora(text_ls,
#                       ptm_name="THUDM/chatglm-6b",
#                       peft_name="shibing624/chatglm-6b-belle-zh-lora"):
#     m = ChatGlmModel('chatglm',
#                      model_name=ptm_name,
#                      peft_name=peft_name,
#                      args={'use_peft': True}, )
#     predict_sentences = [chatglm_generate_prompt(s) for s in text_ls]
#     res = m.predict(predict_sentences)
#     return res
