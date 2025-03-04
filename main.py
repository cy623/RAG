import argparse
from utils import load_dataset
from Agent import LLM
import warnings
from retrieval import retrieval
import os
import time
import subprocess

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"



def QA_process(model, datas, question_string, args):
    for i in range(len(datas)):
        if i < args.num: 
            continue

        try:
            data = datas[i]
            query = data[question_string]
            
            # print("Question:", query)

            # decomposite the query and get the qustion tree
            t = time.time()
            mindmap = model.generate_mindmap(query)
            print("The final query mindmap:", mindmap)
            decompose_time = time.time() - t

            # Extract entities, relations and triples based on question tree
            keys, questions = model.get_key(mindmap)

            triples, descriptions = retrieval(questions, keys)
            print('Triples:', triples)
            print('Descriples:', descriptions)
            # retrival
            # sub_graph  = retrieval(Keys)
            # reasoning and rethink
            t = time.time()
            answer = model.reasoning_and_rethink(mindmap, triples, descriptions)
            print('The final answer: ', answer)
            reasoning_time = time.time() - t

            times = 'The decompose_time: ' + str(decompose_time) + '  ' + 'The reasoning_time: ' + str(reasoning_time)
            file_name = 'RAG-cy/results/' + args.dataset + args.llm + '.txt'
            with open(file_name, 'a') as file:
                file.write(str(i) + '\n' )
                file.write(str(answer) + '\n')
                file.write(times + '\n' )
        except OSError as e:
            print(e)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpot", help="choose the dataset.")
    parser.add_argument("--llm", type=str, default="llama3-8b", choices=['llama2-13b','llama3-8b'], help="choose the llm.")
    parser.add_argument("--num", type=int, default="1", )
    args = parser.parse_args()
    print("Start Running on %s dataset." % args.dataset)

    # load_data
    datas, question_string = load_dataset(args.dataset)
    print('Dataset length: ' + str(len(datas)))

    # get LLM
    model = LLM(args)

    QA_process(model, datas, question_string, args)




if __name__ == "__main__":
    main()