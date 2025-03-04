from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('RAG-cy/bert-base-uncased')
model = BertModel.from_pretrained('RAG-cy/bert-base-uncased').to(device)

def get_sentence_embedding(sentence, tokenizer, model):
    # 将句子编码成BERT所需的格式
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 通过BERT模型进行前向传播，得到输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取[CLS] token对应的向量作为句子的表示
    # outputs[0] 是 (batch_size, sequence_length, hidden_size) 的 tensor
    # 我们取最后一个隐藏状态的第一个 token 对应的向量作为句子的表示
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 取平均表示

def calculate_similarity(sentence1, sentence2):
    # 获取两个句子的嵌入
    embedding1 = get_sentence_embedding(sentence1, tokenizer, model)
    embedding2 = get_sentence_embedding(sentence2, tokenizer, model)
    
    # 计算两个句子嵌入的余弦相似度
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity[0][0]

