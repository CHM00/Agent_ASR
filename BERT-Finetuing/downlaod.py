import os

# 设置国内下载镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型
os.system('huggingface-cli download --resume-download hfl/rbt3 --local-dir ./rbt3')