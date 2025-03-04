import transformers
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
from mindmap import Tree, Node
from queue import Queue
import warnings
import re
warnings.filterwarnings('ignore')
import ast
import time

class LLM:
    def __init__(self, args) -> None:
        self.llm = args.llm
        
        # load_llm
        self.model, self.tokenizer = self.load_llm()

    def load_llm(self):

        if self.llm in ["llama2-13b"]:
            llm_path = "/root/cy-rag/RAG-cy/llm/Llama-2-13b-chat-hf"
            model = AutoModelForCausalLM.from_pretrained(llm_path)
            tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
         
        elif self.llm in ["llama3-8b"]:
            llm_path = "/root/cy-rag/RAG-cy/llm/Llama-3-8B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(llm_path)
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            
        elif self.llm in ["Qwen2.5"]:
            llm_path = "/root/cy-rag/RAG-cy/llm/Qwen2.5"
            model = AutoModelForCausalLM.from_pretrained(llm_path)
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
        

        else:
            raise ValueError("The LLM is Wrong.")
        
        return model, tokenizer
    
  
    
    def run_llm(self, prompt_text):
        model = self.model.eval()  
        if torch.cuda.is_available():
            model = model.to("cuda") 
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(inputs, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()} 
        # print("inputs", inputs["input_ids"])
        with torch.no_grad():  
            outputs = model.generate(
            **inputs,
            max_new_tokens = 1024,
            temperature=1.0,  
            top_p=1.0, 
            do_sample=False,  
            pad_token_id=self.tokenizer.eos_token_id 
        )
    
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        return generated_text
    

    def get_conditions(self, query):
        with open(f'RAG-cy/prompt/prompt.json', 'r', encoding='utf-8') as f:
            json_text = json.load(f)
        for item in tqdm(json_text):
            condition_head = item["Condition_head"]
            prompt_condition = item["Prompt_condition"]
        
        examples = """ 
            Input Q: What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?
            Output: [{"Conditions": "Arnold Schwarzenegger starred in this movie."},
                    {"Conditions": "Arnold Schwarzenegger plays a former New York police officer in the movie."},
                    {"Conditions": "Guns N' Roses performed a promo for the movie."}]
            
            Input Q: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
            Output: [{"Conditions": "The university's main campus is in Lawrence, Kansas."}, 
                    {"Conditions": "The university has branch campuses in the Kansas City metropolitan area."}]
            
            Input Q: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
            Output: [{"Conditions": "No conditions."}]         
        """
        # query = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
        prompt_text = condition_head + prompt_condition + \
                    examples + "\n" + \
                    "The given question Q: " + query + "\n" + \
                    "Output: "               
        
        response = self.run_llm(prompt_text)

        json_part1 = response.split('[', 1)[1]
        json_part2 = json_part1.split(']', 1)[0]
        json_part = '[' + json_part2 +']'
        response = json.loads(json_part)
        # print("response:", response)
        return response
  

    def decompose(self, query):
        with open(f'RAG-cy/prompt/prompt.json', 'r', encoding='utf-8') as f:
            json_text = json.load(f)
        for item in tqdm(json_text):
            decomposition_head = item["Decomposition_head"]
            prompt_decomposition = item["Prompt_decomposition"]

        examples = """
            Input: "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?" 
            Output: [
                    {
                        "Sub-question": "What movie starring Arnold Schwarzenegger as a former New York Police detective is being referred to?",
                        "State": "Continue."
                    },
                    {
                        "Sub-question": "In what year did Guns N Roses perform a promo for the movie mentioned in sub-question #1?",
                        "State": "End."
                    }
                ]
            
            Input: "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
            Output: [
                    {
                        "Sub-question": "Which university has its main campus in Lawrence, Kansas and branch campuses in the Kansas City metropolitan area?",
                        "State": "End."
                    },
                    {
                        "Sub-question": "What is the name of the fight song of the university identified in sub-question #1?",
                        "State": "End."
                    }
                ]
            
                Input: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"
                Output: [
                    {
                        "Sub-question": "Where is the Laleli Mosque located?",
                        "State": "End."
                    },
                    {
                        "Sub-question": "Where is the Esma Sultan Mansion located?",
                        "State": "End."
                    },
                    {
                        "Sub-question": "Are the locations of the Laleli Mosque and the Esma Sultan Mansion in the same neighborhood?",
                        "State": "End."
                    }
                ]
        """
        # query = "What movie starring Arnold Schwarzenegger as a former New York Police detective is being referred to?"
        prompt_text = decomposition_head + prompt_decomposition + \
                    examples + "\n" + \
                    "Input: " + query + "\n" + \
                    "Output: "               
    
        response = self.run_llm(prompt_text)
        # print("decomposition_response:", response)
        json_part1 = response.split('[', 1)[1]
        json_part2 = json_part1.split(']', 1)[0]
        json_part = '[' + json_part2 +']'
        sub_questions = json.loads(json_part)

        return sub_questions
    

    def extraction(self, information):
        with open(f'RAG-cy/prompt/prompt.json', 'r', encoding='utf-8') as f:
            json_text = json.load(f)
        for item in tqdm(json_text):
            extraction_head = item["Extraction_head"]
            prompt_extraction = item["Prompt_extraction"]

        examples = """
            Input: = [
            "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?",
            "What movie starring Arnold Schwarzenegger as a former New York Police detective is being referred to?",
            "In what year did Guns N Roses perform a promo for the movie mentioned in sub-question #1?"
            ]
            Output: [
            {"entity": "(Arnold Schwarzenegger)"},
            {"entity": "(movie)"},
            {"entity": "(year)"},
            {"entity": "(Guns N' Roses)"},
            {"entity": "(New York Police detective)"}
            {"tuples": "(Guns N' Roses, performed a promo for)"},
            {"tuples": "(Arnold Schwarzenegger, starring)"},
            {"tuples": "(former New York police officer, role in)"},
            {"triples": "(Arnold Schwarzenegger, starred in, movie)"},
            {"triples": "(Arnold Schwarzenegger, plays, former New York police officer)"},
            {"triples": "(movie, starring, Arnold Schwarzenegger)"},
            {"triples": "(movie, starring, former New York police officer)"},
            {"triples": "(Guns N' Roses, performed a promo for, movie)"},
            {"triples": "(Guns N' Roses, performed a promo for, year)"}
        ]
        """
         
        prompt_text = extraction_head + prompt_extraction + \
                    examples + "\n" + \
                    "The given input: " + str(information) + "\n" + \
                    "Output: "               

        response = self.run_llm(prompt_text)
        # print("extraction_response:", response)
        json_part1 = response.split('[', 1)[1]
        json_part2 = json_part1.split(']', 1)[0]
        if not json_part2.endswith('}'):
            last_brace_index = json_part2.rfind('}')
            json_part2 = json_part2[:last_brace_index + 1]

        json_part = '[' + json_part2 +']'
        # print(json_part)
        keys = json.loads(json_part)
        return keys
    
    
    def generate_mindmap(self, query):
        self.query = query
        print("Query:", query)

        condition_list = []
        conditions = self.get_conditions(query)
        print("The konwn conditions:", conditions)
        for condition in conditions:
            con = condition['Conditions']
            if con != "No conditions.":
                condition_list.append(con)

        query_Tree = {"0":[query]}
        root = Node(level=0, question=query, state="Continue.")

        N = Queue()
        N.put(root)
        current_level = 0
        level_nodes = []
        the_last_query = None
        the_last_querys = None
        breaks = 0
        t = 0
        while not N.empty() and t < 10:
            t += 1
            nt = N.get()
            if nt.level > current_level: 
                current_level = nt.level
                level_nodes = []

            level_nodes.append(nt)

            if nt.state == "End.": 
                if len(level_nodes) == N.qsize() + 1: 
                    break 
                else:
                    continue
            
            # Decompose
            sub_questions = self.decompose(nt.question)
            print("The sub-questions:", sub_questions)
            # new_nodes = []
            # if the_last_query
            if the_last_querys == sub_questions:
                breaks = 1
            else:
                the_last_querys = sub_questions
                try:
                    for sub_q in sub_questions:
                        if isinstance(sub_q, dict):
                            sub_question = sub_q['Sub-question']
                            if the_last_query == sub_question:
                                state = 'End.'
                                breaks = 1
                            else:
                                state = sub_q['State']
                        else:
                            break
                        the_last_query = sub_question
                        new_node = Node(level=nt.level + 1, question=sub_question, state=state)
                        N.put(new_node)
                        query_Tree.setdefault(str(nt.level + 1), []).append(sub_question)
                except KeyError:
                    continue
            if breaks:
                break
        tree = Tree(condition=condition_list, questions=query_Tree)

        return tree
    
    def get_key(self, mindmap):
        conditions = mindmap.condition
        question_tree = mindmap.questions
        question_and_condition = []

        for value in question_tree.values():
            for _ in value:
                question_and_condition.append(_)
        # print(question_and_condition)
        keys = self.extraction(question_and_condition)
        print("The keys: ", keys)
        return keys, question_and_condition
    

    def rethink(self, knowledge, query, response_all):
        with open(f'RAG-cy/prompt/prompt.json', 'r', encoding='utf-8') as f:
            json_text = json.load(f)
        for item in tqdm(json_text):
            extraction_head = item["Rethink_head"]
            prompt_extraction = item["prompt_rethink"]
    
        last_brace_position = response_all.rfind('{')
        sub_question_answer = '[' + response_all[:last_brace_position] + ']'

        examples = """
            The input question: In which American football game was Malcolm Smith named Most Valuable player?

            The input knowledge: 
            The triples: ['(gridiron football player, Commons category, American football players)', '( , part of, American football team)', '(Malcolm Smith, family name,  )', '(Malcolm Smith, sex or gender,  )', '(American football, practiced by,  )', '(American football, Commons gallery, American football)', '(American football, short name, American football)', '( , sport, American football)', '(Malcolm Arthur Smith, field of work,  )', '(Malcolm Smith, country of citizenship,  )', '( , different from, Malcolm Smith)', '( , field of this occupation, American football)', '( , uses, American football ball)', '(Malcolm Arthur Smith, occupation,  )', '(Malcolm Smith, occupation,  )', '(Malcolm Smith, participant in,  )', '(American football, Commons category, American football)', '(Malcolm Smith, Commons category, Malcolm Smith (American football))', '(American football, subclass of,  )', '(Malcolm Smith, sport,  )']
            The descriptions: ['wild animals under pursuit or taken in hunting', 'specific mathematical model of interactions and payoffs of optimizing agents', 'player of a competitive sport or game', 'segment of a set in tennis, sequence of points played with the same player serving', 'entity vying in some contest (e.g. a person, team, company, creative work)', '2023 early access video game', 'person who plays association football (soccer)', 'British physician and herpetologist (1875–1958)', 'forms of recreational activity, usually physical', 'structured form of play', 'British climber', 'American racing driver', 'player of American football, Canadian football or other gridiron football variants', 'athlete who plays American football', 'wild mammal or bird', 'philosophical concept', 'college basketball player (2012–2015) Lipscomb', 'part of a plant different from the rest', 'Scottish politician (1856-1935)', 'cricketer (1932-2012)', 'German organization', 'American football player', 'form of team game played with an oval ball on a field marked out as a gridiron', 'person who plays a game', 'Papua New Guinean politician', 'British model maker (b.1951)', 'person who plays video games and/or identifies with the gamer identity']

            The subquestion and answers: [
            {"question": 'Who is Malcolm Smith?',
            "answer": "Malcolm Smith is an American football player."},

            {"question": 'In which sport did Malcolm Smith play?',
            "answer": "Malcolm Smith plays American football."},

            {"question": 'Who is Malcolm Smith?',
            "answer": "Malcolm Smith is an American football player."},

            {"question": 'In which sport did Malcolm Smith play?',
            "answer": "Malcolm Smith plays American football."},

            {"question": 'What is the specific American football game played by Malcolm Smith?',
            "answer": "Insufficient information, I don't know."},

            {"question": 'In which American football game did Malcolm Smith play?',
            "answer": "Insufficient information, I don't know."},

            {"question": 'Was Malcolm Smith named Most Valuable Player in the game mentioned in sub-question #1?',
            "answer": "Insufficient information, I don't know."}
            ]
            
            Output: [
                    {"question": 'In which American football game was Malcolm Smith named Most Valuable player?',
                    "answer": "Insufficient information, I don't know."}
                    ]
        """
        # The conditions: ['Malcolm Smith was named Most Valuable Player in an American football game.']
        prompt_text = extraction_head + prompt_extraction + \
                    "The input question: " + str(query) + "\n" + \
                    "The given knowledge: " + knowledge + "\n" + \
                    "The subquestion and answers: " + sub_question_answer + '\n' + \
                    "Output: "    
        # examples + "\n" + \
        
        response = self.run_llm(prompt_text)
        json_list = re.findall(r'\{[^{}]*\}', response)
        response = json_list[-1]

        
        return ast.literal_eval(response.strip())
    

    def reasoning_and_rethink(self, mindmap, triples, descriptions):
       
        with open(f'RAG-cy/prompt/prompt.json', 'r', encoding='utf-8') as f:
            json_text = json.load(f)
        for item in tqdm(json_text):
            extraction_head = item["Reasoning_head"]
            prompt_extraction = item["prompt_reason"]

        examples = """
                    The input questions: [
                    {"question": 'Who is Scott Derrickson?'},
                    {"question": 'Who is Ed Wood?'},
                    {"question": 'Are Scott Derrickson and Ed Wood of the same nationality?'}              
                    ]
                    Output: [
                    {"question": 'Who is Scott Derrickson?',
                    "answer": "Scott Derrickson is an American filmmaker."},
                    {"question": 'Who is Ed Wood?',
                    "answer": "Ed Wood is a 1994 American biographical comedy-drama film directed."},
                    {"question": 'Are Scott Derrickson and Ed Wood of the same nationality?',
                    "answer": "Insufficient information, I don't know."}
                    ]
        """
        conditions = mindmap.condition
        question = mindmap.questions

        q_new = []
        for key in sorted(question.keys(), reverse=True):  # reverse to match the question order
            q_new.extend(question[key])

        q_transformed = [{"question": question} for question in q_new]
       
        
        knowledge = "The triples: " + str(triples) + "\n" + \
                    "The descriptions: " + str(descriptions) + "\n"
        # "The conditions: " + str(conditions) + "\n" + \
        prompt_text = extraction_head + prompt_extraction + \
                    examples + "\n" + \
                    "The given knowledge: " + knowledge + "\n" + \
                    "The input question: " + str(q_transformed) + "\n" + \
                    "Output: "               
        # print(prompt_text)
        response = self.run_llm(prompt_text)

        
        json_list = re.findall(r'\{[^{}]*\}', response)
        response1 = json_list[-1]
        response_all = ''
        for res in json_list:
            response_all = response_all + res + '\n'

        response1 = ast.literal_eval(response1)
        print('The first response is: ', response1)

        examples2 = """
                    The input questions: [
                    {"question": 'Are Scott Derrickson and Ed Wood of the same nationality?'},                   
                    ]
                    Output: [
                    {"question": 'Are Scott Derrickson and Ed Wood of the same nationality?',
                    "answer": "Insufficient information, I don't know."}
                    ]
        """

        extraction_head = """Your task is to answer the questions based on the input knowledge. 
                             Please note that if you cannot infer the answer to the question from the knowledge.\n"""
        prompt_extraction = """
                            Please answer the questions only based on the input information, and do not use your own knowledge. 
                            Please give the answers to the questions in the form of '{\"question\": \"ques1\", \"answer\": \"ans\"}'. \n
        """
        prompt_text2 = extraction_head + prompt_extraction + \
                    "The given knowledge: " + knowledge + "\n" + \
                    "The input question: " + str(q_transformed[-1]) + "\n" + \
                    "Output: "
        
        response2 = self.run_llm(prompt_text2)
       
        json_list = re.findall(r'\{[^{}]*\}', response2)
        response2 = json_list[0]
        response2 = ast.literal_eval(response2)
        print('The second response is: ', response2)

     
        if response2["answer"] == response1["answer"]:
            final_answer = response1
        else:
            final_answer = self.rethink(knowledge, q_transformed[-1], response_all)
        
        return final_answer
    