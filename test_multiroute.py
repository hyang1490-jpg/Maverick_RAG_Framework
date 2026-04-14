"""临时测试脚本：验证三路并行召回结果"""
from icarus_core import getCollection, retrieveCases

query = "我要砸500万做一个社交平台挑战微信"
collection = getCollection()
results = retrieveCases(collection, query)

print("\n" + "=" * 60)
print(f"Query: {query}")
print(f"Total matched (after dedup): {len(results['documents'][0])}")
print("=" * 60)

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
), start=1):
    print(f"\n--- Result {i} ---")
    print(f"  company_name : {meta.get('company_name', 'N/A')}")
    print(f"  archetype    : {meta.get('archetype', 'N/A')}")
    print(f"  industry     : {meta.get('industry', 'N/A')}")
    print(f"  distance     : {dist:.4f}")

print("\n" + "=" * 60)
