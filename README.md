# llmeval

下载：

```shell
pip install -r requirements.txt
pip install -e .

# 安装textgen
# pip install -U textgen
git clone https://github.com/shibing624/textgen.git
cd textgen
python setup.py install
cd ..
```



参数：

```shell
llmeval [-h] -i INPUT : 输入的question，文本文件
			 -o OUTDIR: 输出目录，保存评估结果
			 --infer-names: 推理函数的唯一标识，指定多个需要以空格分隔
			 			（推理函数需要用llmeval.infer.register_llm_infer注册）
             [-c --conf: 指定配置文件，用于更新推理需要的权重]
			 [--base-model: 用chatgpt输出参考答案（默认gpt-3.5-turbo）]
             [--review-model: 审查模型名,用于对llm和chatgpt输出打分（gpt-3.5-turbo/gpt-4）] 
             [--tozh: 指定该参数后会将review模型的输出转为中文 ]
             [--max-tokens: openai gpt模型输出的最大token数] 
             [--temperature 文本生成的温度，小准确，大多样性高（默认0.2）]
             [--max-api-retry 最大重调api次数，失败了会再次调用（默认5）] 
             [--req-time-gap openai请求等待的时间（10）]
             [--llm-ckpt: llm的权重 ] （暂无用）
             [--peft-ckpt: llm peft的权重] （暂无用）

```

其中-i -o --infer-names是必填参数，输出目录中的文件格式如下：

```shell
outdir/
# chatgpt和llm输出（answer_${infer_name}.jsonl）
answer_gpt35.jsonl  
answer_hello.jsonl    
# review模型的打分（review_${infer_name}.$suffix）
review_hello.jsonl  
review_hello.xlsx
# 合并outdir下所有infer_names结果到excel
merged_llmeval.xlsx

# 注：如果输出目录有对应的文件，会跳过该阶段运行
```



运行：

```shell
# 不运行llm
llmeval -i out/question.txt  -o out --infer-names hello

# 运行llm,需要将llmeval.infer.textgen_infer中的chatglm_6b推理函数开启
llmeval -i  examples/data/llm_benchmark_tiny.txt  -o output --infer-names chatglm_6b

# 指定配置文件修改权重
llmeval -i  examples/data/llm_benchmark_tiny.txt  -o output --infer-names chatglm_6b -c config.yaml

```



项目目录：

```shell
llmeval/
	llmeval/
        baseline.py   # chatgpt推理模块
        infer/  	  # llm推理模块
        review.py	  # 打分模块
        file_io.py    # 文件的读写
        options.py    # 命令行参数
	llmeval_cli/
        run.py  	  # 运行入口
    requirements.txt  # 依赖
    setup.py		  # 安装
    config.yaml 	  # 权重配置文件
```



bug:

1. ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'

   ```shell
   pip install chardet charset-normalizer==2.1.0
   # 参考：https://stackoverflow.com/questions/74535380/importerror-cannot-import-name-common-safe-ascii-characters-from-charset-nor
   ```

2. PyTorch Error loading "\lib\site-packages\torch\lib\shm.dll" or one of its dependencies

   ```shell
   conda install cudatoolkit
   # 参考：https://stackoverflow.com/questions/74594256/pytorch-error-loading-lib-site-packages-torch-lib-shm-dll-or-one-of-its-depen
   ```

3. RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback): cannot import name 'UnencryptedCookieSessionFactoryConfig' from 'pyramid.session' (unknown location)   

   ```shell
   git clone https://github.com/NVIDIA/apex
   cd apex
   python setup.py install
   ```

4. xx