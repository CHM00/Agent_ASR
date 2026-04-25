from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import Neo4jError


NEO4J_URI = "bolt://47.113.202.238:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Zchm0903"


class Neo4jConnectionTest:
    def __init__(self, uri, user, password):
        """初始化连接"""
        self._driver = None
        try:
            # 创建驱动对象
            self._driver = GraphDatabase.driver(
                uri,
                auth=basic_auth(user, password)
            )
            # 验证连接是否有效
            self._driver.verify_connectivity()
            print("Neo4j 连接成功！")
        except Neo4jError as e:
            print(f"Neo4j 连接失败：{e}")
            raise

    def run_test_query(self):
        """运行测试查询，验证数据库可操作"""
        if not self._driver:
            print("驱动未初始化，无法执行查询")
            return

        try:
            # 执行简单的查询（创建一个测试节点并查询）
            with self._driver.session(database="neo4j") as session:
                # 创建测试节点
                session.run("CREATE (:TestNode {name: 'ConnectionTest', time: datetime()})")
                # 查询所有TestNode节点
                result = session.run("MATCH (n:TestNode) RETURN n.name, n.time")

                # 打印查询结果
                print("测试查询结果：")
                for record in result:
                    print(f"节点名称：{record['n.name']}，创建时间：{record['n.time']}")

                # 清理测试数据（可选，避免残留）
                session.run("MATCH (n:TestNode) DELETE n")
                print(" 测试查询执行成功，已清理测试数据")
        except Neo4jError as e:
            print(f" 查询执行失败：{e}")

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            print(" Neo4j 连接已关闭 ")


# ===================== 运行测试 =====================
if __name__ == "__main__":
    conn = None
    try:
        # 初始化连接
        conn = Neo4jConnectionTest(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        # 运行测试查询
        conn.run_test_query()
    except Exception as e:
        print(f" 程序执行异常：{e}")
    finally:
        # 确保连接关闭
        if conn:
            conn.close()