import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class KnowledgeGraph:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.environ.get("NEO4J_USER", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print(" Neo4j 连接成功")
            self.init_constraints()
        except Exception as e:
            print(f" 连接失败: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def init_constraints(self):
        queries = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            # 这里的 Entity 泛指所有提取出的目标，如 Food, Person 等
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    # 插入关系  绑定Milvus ID
    def upsert_relation(self, user_id, relation, target, target_type, milvus_id):
        """
        插入图谱，并将 Milvus 的 ID 记录在关系上，便于联动删除。
        结构: (User)-[:RELATION {mid: 12345}]->(Entity)
        """
        cypher = f"""
        MERGE (u:User {{id: $user_id}})
        MERGE (t:{target_type} {{name: $target}})
        MERGE (u)-[r:{relation.upper()}]->(t)
        SET r.mid = $milvus_id
        SET r.timestamp = timestamp()
        """
        with self.driver.session() as session:
            session.run(cypher,
                        user_id=user_id,
                        target=target,
                        milvus_id=milvus_id)
            print(f" 关系已建立: {user_id} -[{relation}]-> {target} (绑定的MilvusID: {milvus_id})")

    #根据 Milvus ID 删除关系
    def delete_relation_by_mid(self, milvus_id):
        """
        当 Milvus 删除了某条记忆，图谱也通过 mid 找到对应边并删除
        """
        cypher = """
        MATCH (u:User)-[r]->(t)
        WHERE r.mid = $milvus_id
        DELETE r
        """
        with self.driver.session() as session:
            session.run(cypher, milvus_id=milvus_id)
            print(f" 已联动删除关系 (MilvusID: {milvus_id})")

    # 图谱检索
    def search_user_graph(self, user_id):
        """查询用户的一阶关系"""
        cypher = """
        MATCH (u:User {id: $user_id})-[r]->(t)
        RETURN type(r) as relation, t.name as target
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(cypher, user_id=user_id):
                results.append(f"{record['relation']} {record['target']}")
        return results

    def clear_database(self):
        """清空所有数据和索引"""
        try:
            with self.driver.session() as session:
                # 删除所有节点和关系
                print("正在删除所有图谱数据...")
                session.run("MATCH (n) DETACH DELETE n")

                # 删除旧的向量索引
                print("正在删除旧索引...")
                session.run("DROP INDEX entity_embedding_index IF EXISTS")

                # 删除其他约束
                session.run("DROP CONSTRAINT user_id_unique IF EXISTS")
                session.run("DROP CONSTRAINT food_name_unique IF EXISTS")

            print(" 数据库已清空，索引已移除。")
        except Exception as e:
            print(f"清空失败: {e}")

if __name__=="__main__":
    kg = KnowledgeGraph()
    kg.connect()
    kg.clear_database()