import os
import asyncio
import time
import json, random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# 语音识别相关库
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# LLM API相关库
from openai import AsyncOpenAI
from dotenv import load_dotenv
# import tqdm
# 与MilVus向量库有关
import pandas as pd
from langchain_core.documents import Document
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import configparser
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import threading


# 加载环境变量(.env文件)
load_dotenv()

class MilvusClass:
    def __init__(self):
        self.MILVUS_URI = os.environ.get("URL")  # Milvus 地址
        self.MILVUS_TOKEN = os.environ.get("Token")  # Milvus Token
        self.ARK_API_KEY = os.environ.get("ARK_API_KEY")  # 填入你的 API Key
        self.ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.embedding_url = self.ARK_BASE_URL + "/embeddings"
        self.LLM_MODEL = "deepseek-ai/DeepSeek-V3"  # 你的推理模型 ID
        self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"  # 你的 Embedding 模型 ID
        self.conn = "link"
        self.embedding_dim = 2560
        self.food_name = "Food_List"
        self.mem_name = "User_Memory"
        self.food_collection = None
        self.memory_collection = None

    def connect_milvus(self):
        """连接 Milvus 数据库并初始化记忆集合"""
        try:
            connections.connect(alias="link", uri=self.MILVUS_URI, token=self.MILVUS_TOKEN)
            print("✅ [Brain] Milvus 连接成功")

            self.init_food_collection()

            # 2. 初始化用户记忆集合 'User_Memory'
            self.init_memory_collection()

        except Exception as e:
            print(f"[Brain] Milvus 连接失败或集合初始化错: {e}")
            self.food_collection = None
            self.memory_collection = None

    def init_food_collection(self):
        try:
            # 检查集合是否存在
            if utility.has_collection(self.food_name, using=self.conn):
                print(f"集合 {self.food_name} 存在。")
                self.food_collection = Collection(name=self.food_name, using=self.conn)
                self.food_collection.load()
                print(f"集合字段: {[field.name for field in self.food_collection.schema.fields]}")
            else:
                print(f"集合 {self.food_name} 不存在，准备创建新集合。")
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                    FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="category_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="cate_1_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="cate_2_name", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="cate_3_name", dtype=DataType.VARCHAR, max_length=255)
                ]

                # 创建 Schema
                schema = CollectionSchema(
                    fields=fields,
                    description="data Base Vectors",
                    enable_dynamic_field=False
                )

                # 创建集合
                self.food_collection = Collection(name=self.food_name, schema=schema, using=self.conn)
                print(f"集合 {self.food_name} 创建成功。")

                # 创建索引
                index_params = {
                    "metric_type": "IP",
                    "index_type": "FLAT",
                    "params": {"M": 16, "efConstruction": 200}
                }
                self.food_collection.create_index(field_name="vector", index_params=index_params)
                print("索引创建完成。")

                # 加载集合到内存
                self.food_collection.load()
                print(f"[Food] 新建食材库: {self.food_name}")

        except Exception as e:
            print(f"Milvus 操作失败: {e}")

    def init_memory_collection(self):
        """创建或加载用户记忆集合 (user_id 字段)"""
        if utility.has_collection(self.mem_name, using="link"):
            self.memory_collection = Collection(self.mem_name, using="link")
            self.memory_collection.load()
            print(f"[Memory] 加载长期记忆库: {self.mem_name}")
        else:
            # 定义 Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),  # 新增用户id
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields, "用户长期画像记忆")
            self.memory_collection = Collection(self.mem_name, schema, using="link")

            index_params = {"metric_type": "IP", "index_type": "FLAT", "params": {"M": 8, "efConstruction": 64}}
            self.memory_collection.create_index("vector", index_params)

            # 为 user_id 创建标量索引，加速过滤
            self.memory_collection.create_index("user_id", {"index_type": "Trie"})

            self.memory_collection.load()
            print(f"[Memory] 新建长期记忆库(多用户版): {self.mem_name}")

    def search_memory(self, query_text, user_id, top_k=3):  # <--- 新增 user_id 参数
        """检索特定用户的记忆"""
        if not self.memory_collection: return []

        vec = self.embedding(query_text)
        if not vec: return []

        search_params = {"metric_type": "IP", "params": {"ef": 64}}

        # 核心修改: 增加 expr 表达式，只搜索该用户的记忆
        filter_expr = f'user_id == "{user_id}"'

        try:
            res = self.memory_collection.search(
                data=[vec],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["text", "id", "user_id"]
            )

            results = []
            if res and res[0]:
                for hit in res[0]:
                    results.append({
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "score": hit.distance
                    })
            return results
        except Exception as e:
            print(f"Milvus 检索失败: {e}")
            return []

    def insert_memory(self, text, user_id):
        """新增: 显式插入记忆的方法"""
        vec = self.embedding(text)
        if vec:
            import time
            # 数据顺序必须与 Schema 定义一致: [user_id, vector, text, timestamp] (id是自动生成的)
            # 注意 pymilvus insert 的顺序通常是按列插入
            data = [
                [user_id],  # user_id 列
                [vec],  # vector 列
                [text],  # text 列
                [int(time.time())]  # timestamp 列
            ]
            self.memory_collection.insert(data)
            print(f"[Memory] 已写入 {user_id} 的记忆: {text}")


    def embedding(self, text):
        payload = {
            "model": self.EMBEDDING_MODEL,
            "input": f"{text}",
        }
        headers = {
            "Authorization": f"Bearer {self.ARK_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.embedding_url, json=payload, headers=headers)
            # response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            embedding_vec = result['data'][0]["embedding"]
            print(len(embedding_vec))
            # print(result)
            return embedding_vec
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP 错误发生: {http_err}")
        except Exception as err:
            print(f"其他错误发生: {err}")

    def deleteMilvus(self, collection_name="MilVus_test"):
        # 检查集合是否存在
        try:
            if utility.has_collection(collection_name, using=self.conn):
                print(f"集合 {collection_name} 存在。")
                collection = Collection(name=collection_name, using=self.conn)
                print(f"集合字段: {[field.name for field in collection.schema.fields]}")
                collection.drop()
                print(f"Milvus 删除集合{collection_name} 成功")
        except Exception as e:
            print(f"Milvus 删除集合失败: {e}")


    def batch_embedding(self, texts: List[str], batch_size: int = 50, max_workers: int = 4) -> List[List[float]]:
        """多线程批获取 embedding"""
        all_embeddings = [None] * len(texts)  # 预分配，保持顺序
        lock = threading.Lock()

        # 分割成批次
        batches = [(i, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]

        def process_batch(batch_info):
            start_idx, batch_texts = batch_info
            payload = {
                "model": self.EMBEDDING_MODEL,
                "input": batch_texts,
            }
            headers = {
                "Authorization": f"Bearer {self.ARK_API_KEY}",
                "Content-Type": "application/json"
            }

            try:
                response = requests.post(self.embedding_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                batch_embeddings = [item["embedding"] for item in sorted(result['data'], key=lambda x: x['index'])]
                return start_idx, batch_embeddings
            except Exception as e:
                print(f"批量 embedding 失败 (idx={start_idx}): {e}")
                return start_idx, [[0.0] * self.embedding_dim] * len(batch_texts)

        # 多线程并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}

            for future in tqdm(as_completed(futures), total=len(batches), desc="Embedding 并发处理"):
                start_idx, embeddings = future.result()
                # 按原始位置存入结果
                for j, emb in enumerate(embeddings):
                    all_embeddings[start_idx + j] = emb

        return all_embeddings

    def Batch_insert_food(self, file_path, one_bulk=100, embedding_batch=100):
        df = pd.read_csv(file_path, sep='\s+')

        # 插入数据
        data_to_insert = []
        valid_rows = []
        texts = []
        for index, row in df.iterrows():
            item_name = str(row['item_name'])
            category_name = str(row['category_name'])
            cate_1_name = str(row['cate_1_name'])
            cate_2_name = str(row['cate_2_name'])
            cate_3_name = str(row['cate_3_name'])

            non_empty_strings = [s for s in [item_name, category_name, cate_1_name, cate_2_name, cate_3_name] if s]
            text = ''.join(non_empty_strings)
            # 拼接文本信息
            # text = item_name + category_name + cate_1_name + cate_2_name + cate_3_name
            if text:
                texts.append(text)
                valid_rows.append({
                    'item_name': item_name,
                    'category_name': category_name,
                    'cate_1_name': cate_1_name,
                    'cate_2_name': cate_2_name,
                    'cate_3_name': cate_3_name
                })
        # 2. 批量获取 embedding
        print(f"\n 开始批量 Embedding ({len(texts)} 条数据)...")
        # embeddings = self.batch_embedding(texts, batch_size=embedding_batch)
        embeddings = self.batch_embedding(texts, batch_size=100, max_workers=6)

        # 3. 组装数据并批量插入
        print("\n 插入 Milvus...")
        data_to_insert = []
        for i, (emb, row_data) in enumerate(zip(embeddings, valid_rows)):
            data_to_insert.append([
                emb,
                row_data['item_name'],
                row_data['category_name'],
                row_data['cate_1_name'],
                row_data['cate_2_name'],
                row_data['cate_3_name']
            ])

        # 批量插入
        for i in tqdm(range(0, len(data_to_insert), one_bulk), desc="Milvus 插入"):
            batch_entities = list(map(list, zip(*data_to_insert[i:i + one_bulk])))
            try:
                self.food_collection.insert(batch_entities)
            except Exception as e:
                print(f"文档插入 Milvus 失败: {e}")

        self.food_collection.flush()
        print(f"完成! 共插入 {len(data_to_insert)} 条数据")


    # 删除
    def delete_memory_by_ids(self, id_list, user_id):
        """根据 ID 列表和 user_id 安全删除记忆"""
        if not self.memory_collection or not id_list: return

        try:
            # 同时校验 id 和 user_id, 只有当 ID 在列表中，且该 ID 属于指定 user_id 时才删除
            expr = f"id in {id_list} and user_id == '{user_id}'"

            self.memory_collection.delete(expr)
            self.memory_collection.flush()  # 确保删除立即生效
            print(f"[Milvus] 已安全删除用户 {user_id} 的记忆 ID: {id_list}")

        except Exception as e:
            print(f"Milvus 删除失败: {e}")

if __name__ == '__main__':
    milvus_instance = MilvusClass()
    milvus_instance.connect_milvus()
    # milvus_instance.deleteMilvus("User_Memory")
    # milvus_instance.Batch_insert_food(r"D:\ASR-LLM-TTS-master\ASR-LLM-TTS-master\food_category.txt", one_bulk=100)