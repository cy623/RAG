import json


def load_dataset(dataset_name):
    if dataset_name == 'hotpot':
        with open('RAG-cy/data/Hotpot/processed_data/hotpot_dev_new.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_key = 'question'

    elif dataset_name == 'CWQ':
        with open('RAG-cy/data/CWQ/ComplexWebQuestions_dev.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_key = 'question'
    elif dataset_name == 'WebQSP':
        with open('RAG-cy/data/WebQSP/data/WebQSP.train.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_key = 'RawQuestion'
        # print(datas[Questions])
        return datas['Questions'], question_key
    
    return datas, question_key



