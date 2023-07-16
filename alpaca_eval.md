先使用llmeval对不同的模型进行推理，需要指定alpaca-eval参数生成alpaca_eval需要的数据格式。然后再用alpaca_eval的make_leaderboard

环境，推理

## 1.准备环境

虚拟环境：

```shell
conda create -n eval python=3.11.3
conda activate eval
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.28.1
```

安装alpaca_eval：

```shell
pip install alpaca-eval
```

设置openai环境变量：

```shell
export OPENAI_API_KEY=<your_api_key>
```

安装llmeval：

```shell
git clone https://github.com/jiaohuix/llmeval
cd llmeval
pip install -e .
```

## 2.数据、模型准备

2.1 数据准备：

准备纯文本的问题，比如examples/data/llm_benchmark_tiny.txt：

```
我能用lightning数据线给安卓手机充电吗？
为什么天空是蓝色的？
如何做披萨？
```

2.1 准备模型：

​		llmeval支持自定义推理函数，可以直接使用huggingface上的模型进行推理（跑通后将推理函数用装饰器register_llm_infer注册唯一的推理名字）。模型的推理代码写在llmeval/infer下的任意python文件，会被自动扫描，多个问题text_ls输入到推理函数中，返回答案列表res。

​		下面的例子将baichuan_7b改为llmeval需要的推理函数：

​		首先运行hf的代码，下载完整权重并完成推理。

```python
#https://huggingface.co/baichuan-inc/Baichuan-7B
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

​		然后创建推理代码： llmeval/infer/baichuan_infer.py，内容为：

```shell
from llmeval.infer import register_llm_infer
from transformers import AutoModelForCausalLM, AutoTokenizer


@register_llm_infer("baichuan_7b")
def infer_baichuan_7b(text_ls,
              ptm_name="baichuan-inc/Baichuan-7B",
              peft_name=None):
    '''百川7b的推理函数。

    Args:
        text_ls:list[str] 包含若干文本问题的列表。
        ptm_name:str   模型的标识，用来加载模型和权重。
        peft_name:str   lora或额外参数的加载路径，default=None。

    Returns:list[str]   包含若干答案的列表，由模型推理得到。

    '''
    print(f"ptm_name:{ptm_name}")
    print(f"peft_name:{peft_name}")
    res = []
    tokenizer = AutoTokenizer.from_pretrained(ptm_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ptm_name, device_map="auto", trust_remote_code=True)
    for text in text_ls:
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to('cuda:0')
        pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        res.append(ans)
    return res
```

​	

## 3.模型推理

推理单个模型(chatglm_6b)：

```shell
llmeval -i examples/data/llm_benchmark_tiny.txt -o output --infer-names chatglm_6b --alpaca-eval
```

推理多个模型(chatglm_6b baichuan_7b)，以空格分隔：

```shell
llmeval -i examples/data/llm_benchmark_tiny.txt -o output --infer-names chatglm_6b baichuan_7b --alpaca-eval
```

--alpaca-eval参数生成alpaca_eval库需要的数据格式，不管是单个模型还是多个模型，全输出到输出目录-o下的**output/output.json**，数据格式如下：

```
[
  {
      "instruction": "我能用lightning数据线给安卓手机充电吗？",
    "input": "",
    "output": "Lightning数据线可以用于给安卓手机充电，但是需要使用支持 lightning 接口的充电器。\n\n使用支持 lightning 接口的充电器可以给任何支持 lightning 接口的设备和充电，包括安卓智能手机。不过，由于 lightning 接口和充电器的规格不同，因此需要使用正确的充电器才能确保充电效果和充电安全。",
    "generator": "chatglm_6b_api",
    "dataset": "helpful_base",
    "datasplit": "eval"
  },
]
```

​		其中每个模型对所有问题依次推理输出，其中generator字段决定不同模型。

​		如果需要制作榜单，还需要gpt35的输出，路径为 **output/output_gpt35.json**，内容为：

```shell
[
  {
    "instruction": "我能用lightning数据线给安卓手机充电吗？",
    "input": "",
    "output": "不可以，因为lightning数据线是苹果公司专有的充电和数据传输接口，只能用于苹果设备。如果你有安卓手机，你需要使用与之兼容的充电和数据传输接口，例如micro-USB或USB-C。",
    "generator": "gpt-3.5-turbo",
    "dataset": "helpful_base",
    "datasplit": "eval"
  },
]
```



## 4.评估、制作榜单

​	输入多个模型的推理输出，以及作为参考的gpt35输出，指定评估模型为chatgpt_fn，然后榜单结果输出到board.csv。

```shell
alpaca_eval make_leaderboard \
  --leaderboard_path  board.csv \
  --reference_outputs 'output/output_gpt35.json'  \
  --all_model_outputs 'output/output.json' \
  --annotators_config chatgpt_fn

#leaderboard_path：保存排行榜的路径。排行榜将保存为 csv 文件，如果已经存在，则会追加。

#all_model_outputs：要添加到排行榜的所有模型的输出的 json 路径（作为单个文件或通配多个文件）。每个字典应包含提示中格式化的键 (instruction和) 以及包含当前模型名称的列。作为示例，请参阅此文件。outputgenerator

#reference_outputs参考模型输出的路径。每个字典应包含提示中格式化的键 (instruction和)。output默认情况下，参考输出是 AlpacaEval 集上的 003 输出。

#annotators_config：注释器配置文件的路径。默认为alpaca_eval_gpt4.
```

​		其中annotators_config指定评估的模型，可以为alpaca_eval_gpt4，claude，chatgpt_fn，默认为alpaca_eval_gpt4。

2023/7/16

