import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from fast_multimodal_decision import FastMultimodalDecisionEngine

def run_tests():
    engine = FastMultimodalDecisionEngine()
    
    test_cases = [
        {
            "name": "多模态纯色识别 - 红色方块",
            "image": Image.new("RGB", (64, 64), color="red"),
            "question": "这张图片的主要颜色是什么？",
            "candidates": ["红色", "蓝色", "绿色", "黄色"]
        },
        {
            "name": "多模态纯色识别 - 蓝色方块",
            "image": Image.new("RGB", (64, 64), color="blue"),
            "question": "图中展示的是哪一种主要色调？",
            "candidates": ["绿色", "红色", "蓝色", "黑色"]
        },
        {
            "name": "多模态纯色识别 - 绿色方块",
            "image": Image.new("RGB", (64, 64), color="green"),
            "question": "请选出与输入图像背景匹配的颜色：",
            "candidates": ["紫色", "黄色", "粉色", "绿色"]
        },
        {
            "name": "纯文本通用常识决策 - 鲁迅作品",
            "image": None,
            "question": "《狂人日记》是中国现代文学史上第一篇白话短篇小说，其作者是谁？",
            "candidates": ["茅盾", "鲁迅", "老舍", "巴金", "郁达夫"]
        }
    ]
    
    print("\n" + "="*60)
    print("开始执行低延迟多模态决策评测 (Mugi Fast Multimodal Engine)")
    print("="*60)
    
    for case in test_cases:
        res = engine.predict_fast(
            image=case["image"],
            question=case["question"],
            candidates=case["candidates"]
        )
        print(f"\n[测试项] {case['name']}")
        print(f"  问题: {case['question']}")
        print(f"  前缀 Token 数: {res['prefix_tokens']} | 选项数: {res['candidates_count']}")
        print(f"  预测最佳选项: 【{res['best_candidate']}】 (置信度: {res['probabilities'][res['best_index']]*100:.2f}%)")
        print(f"  分支总延迟: {res['latency_ms']:.2f} ms")
        print("  候选明细打分:")
        for cand, score, prob in zip(case["candidates"], res["scores"], res["probabilities"]):
            bar = "█" * int(prob * 20)
            print(f"    - {cand:<6}: 分数={score:+6.3f} | 概率={prob*100:5.2f}% | {bar}")
            
    print("\n" + "="*60)
    print("所有测试顺利完成！")
    print("="*60)

if __name__ == "__main__":
    run_tests()
