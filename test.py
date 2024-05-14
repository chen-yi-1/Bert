import json  # 导入 json 库用于处理 JSON 格式数据
import torch  # 导入 PyTorch 库
from torch.utils.data import DataLoader, Dataset  # 导入 PyTorch 的 DataLoader 和 Dataset 类
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    AutoConfig  # 导入 Hugging Face 的 Transformers 库中的相关类和函数
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import os

import sys


def test_loop(dataloader, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()  # 将模型设置为评估模式，不进行梯度更新
    correct_predictions = 0  # 初始化正确预测数量
    total_predictions = 0  # 初始化总预测数量
    with torch.no_grad():  # 不需要计算梯度
        output_file = os.path.join(output_dir, "predictions.jsonl")  # 定义输出文件路径
        with open(output_file, 'w', encoding='utf-8') as f:  # 打开输出文件
            for batch in tqdm(dataloader):  # 遍历数据加载器中的每个批次
                ids = batch['id']  # 获取批次中的 ID
                text1s = batch['text1']  # 获取批次中的文本1
                text2s = batch['text2']  # 获取批次中的文本2
                labels = batch.get('label', None)  # 获取批次中的标签（如果存在）
                inputs = tokenizer(text=text1s, text_pair=text2s, padding=True, return_tensors="pt")  # 使用 tokenizer 处理文本并转换为模型输入格式
                inputs = {key: inputs[key].to(device) for key in inputs}  # 将输入数据移动到指定设备

                outputs = model(**inputs)  # 前向传播
                logits = outputs.logits  # 获取模型输出的 logits
                probs = torch.softmax(logits, dim=1)  # 对 logits 进行 softmax 操作得到概率
                confidence_scores = probs[:, 1].tolist()  # 获取文本2是人类作者的置信度分数

                for id_, confidence_score in zip(ids, confidence_scores):  # 遍历每个样本的 ID 和对应的置信度分数
                    f.write(json.dumps({"id": id_, "is_human": confidence_score}) + '\n')  # 将结果写入输出文件

                if labels is not None:  # 如果存在标签
                    correct_predictions += torch.sum((probs.argmax(dim=1) == labels.to(device))).item()  # 计算正确预测数量
                    total_predictions += len(labels)  # 计算总预测数量

        if labels is not None:  # 如果存在标签
            accuracy = correct_predictions / total_predictions  # 计算准确率
            print(f"Accuracy: {accuracy:.2f}")  # 打印准确率


def collate_fn(batch_samples):
    batch_ids = []  # 存储批次中的 ID
    batch_text1s = []  # 存储批次中的文本1
    batch_text2s = []  # 存储批次中的文本2
    batch_labels = []  # 存储批次中的标签
    for sample in batch_samples:  # 遍历批次中的每个样本
        batch_ids.append(sample['id'])  # 将 ID 添加到列表中
        batch_text1s.append(sample['text1'])  # 将文本1添加到列表中
        batch_text2s.append(sample['text2'])  # 将文本2添加到列表中
        if 'label' in sample:  # 如果样本中存在标签
            batch_labels.append(sample['label'])  # 将标签添加到列表中
    if batch_labels:  # 如果存在标签
        return {"id": batch_ids, "text1": batch_text1s, "text2": batch_text2s, "label": torch.tensor(batch_labels)}  # 返回批次的 ID、文本1、文本2和标签的张量形式
    else:
        return {"id": batch_ids, "text1": batch_text1s, "text2": batch_text2s}  # 返回批次的 ID、文本1和文本2
class MyData(Dataset):
    def __init__(self, input_file):
        self.data = self._read_json(input_file)  # 读取输入文件中的数据

    def _read_json(self, input_file):
        lines = []  # 存储读取的数据行
        with open(input_file, "r", encoding="utf-8") as f:  # 打开输入文件
            for line in f:  # 遍历文件中的每一行
                data = json.loads(line)  # 将 JSON 字符串解析为 Python 对象
                lines.append(data)  # 将解析后的数据添加到列表中
        return lines  # 返回所有数据行

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.data[idx]  # 获取指定索引位置的数据


if __name__ == '__main__':
    # --------------------------参数设置------------------------------------------
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    batch_size = 12
    checkpoint = "/t-ng/Bert/bert_model/MyModel"
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    # --------------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # 加载预训练的 tokenizer
    config = AutoConfig.from_pretrained(checkpoint)  # 加载预训练模型配置
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)  # 加载预训练的分类模型

    print("Reading data...")  # 打印提示信息
    test_data = MyData(input_path)  # 加载测试数据集
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # 创建数据加载器
    print("Data reading complete.")  # 打印提示信息

    print("Starting inference...")  # 打印提示信息
    test_loop(test_dataloader, model, output_dir)  # 进行推断
    print("Inference complete.")  # 打印提示信息
