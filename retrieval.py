import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial 
import ujson as json
import nltk
from nltk.corpus import wordnet as wn
import gensim
from bertmodel import calculate_similarity
from collections import defaultdict

workers = 64
chunksize = 64



def jsonl_generator(fname):
    """ Returns generator for jsonl file """
    for line in open(fname, 'r'):
        line = line.strip()
        if len(line) < 3:
            d = {}
        elif line[len(line) - 1] == ',':
            d = json.loads(line[:len(line) - 1])
        else:
            d = json.loads(line)
        yield d


def get_batch_files(fdir):
    """ Returns paths to files in fdir """
    filenames = os.listdir(fdir)
    filenames = [os.path.join(fdir, f) for f in filenames]
    print(f"Fetched {len(filenames)} files from {fdir}")
    return filenames


def filtering_func(target_name, filename, value):
    filtered = []
    for item in jsonl_generator(filename):
        if item[value] in target_name:
            filtered.append(item)
    return filtered


def deduplication(data):
    entities = set()
    tuples = set()
    triples = set()
    unique_data = []

    for item in data:
        if 'entity' in item:
            if item['entity'] not in entities:
                unique_data.append(item)
                entities.add(item['entity'])
        elif 'triples' in item:
            if item['triples'] not in triples:
                unique_data.append(item)
                triples.add(item['triples'])
        elif 'tuples' in item:
            if item['tuples'] not in tuples:
                unique_data.append(item)
                tuples.add(item['tuples'])
    return unique_data


def get_entity_relation(keys):
    entities = set()
    relations = set()
    for item in keys:
        if 'entity' in item:
            if item['entity'] not in entities:
                e = item['entity'].split('(', 1)[1]
                e = e.split(')', 1)[0]
                entities.add(e)
        elif 'triples' in item:
            s = item['triples']
            e1 = s.split('(', 1)[1]
            e2 = e1.split(',', 2)[2]
            r1 = e1.split(',')[1].strip()
            e1 = e1.split(',', 1)[0].strip()
            e2 = e2.split(')', 1)[0].strip()
            if e1 not in entities:
                entities.add(e1)
            elif e2 not in entities:
                entities.add(e2)
            elif r1 not in relations:
                relations.add(r1)
        elif 'tuples' in item:
            s = item['tuples']
            e1 = s.split('(', 1)[1]
            r1 = e1.split(',', 1)[1]
            e1 = e1.split(',', 1)[0].strip()
            r1 = r1.split(')', 1)[0].strip()
            if e1 not in entities:
                entities.add(e1)
            elif r1 not in relations:
                relations.add(r1)

    return entities, relations



################## Get qid ################################

def filtering_func_for_qid(target_name, filename, value):
    filtered = []
    entity_and_alias = []
    for item in jsonl_generator(filename):
        entity_name = item[value]
        alias = item['alias']
        if entity_name in target_name or any(name in target_name for name in alias):
            filtered.append(item)
            entity_and_alias.append({
            "entity": entity_name,
            "alias": alias
        })
            
    return filtered, entity_and_alias


def get_entity_id(entity):  
    data_path = "RAG-cy/wikidata/process/new_labels"
    table_files = get_batch_files(data_path)
    pool = Pool(processes = workers)
    value = 'label'
    filtered = []
    entity_and_alias = []
    # for e in entity:
    for output1, output2 in tqdm(
            pool.imap_unordered(
                partial(filtering_func_for_qid, entity, value=value), table_files, chunksize=chunksize), 
            total=len(table_files)
        ):
            filtered.extend(output1)
            entity_and_alias.extend(output2)
    return filtered, entity_and_alias

################################################




################## Get description for entity ##############################
def get_description(qid):
    qid_values = [item['qid'] for item in qid]
    # print(qid_values)
    data_path = "RAG-cy/wikidata/process/descriptions"
    table_files = get_batch_files(data_path) 
    pool = Pool(processes = workers)
    value = 'qid'
    description = []
    for output in tqdm(
            pool.imap_unordered(
                partial(filtering_func, qid_values, value=value), table_files, chunksize=chunksize), 
            total=len(table_files)
        ):
            description.extend(output)
    return description

###############################################################


###################### Pruning description####################

def Pruning_description(description, question):
    threshold = 0.5
    qid_new = []
    description_new = []
    for desc_item in description:
        description = desc_item['description']
        for ques in question:
            try:
                # similarity = model.similarity(description, ques)
                similarity = calculate_similarity(description, ques)
                # print(similarity)
                if similarity > threshold:
                    qid_new.append(desc_item['qid'])
                    description_new.append(desc_item)
                    break
            except KeyError:
                continue

    return qid_new, description_new
#################################################################


###################### Get entity relations #######################

def filtering_relation(target_name, filename, value):
    filtered = []
    # qid_number = len(target_name)
    num = 0
    name  = ''
    for item in jsonl_generator(filename):
        
        if item[value] in target_name:
            filtered.append(item)
            if item[value] != name:
                num += 1
            name = item[value]
            
    return filtered, num

def get_entity_rels(qid):
    qid_values = [item['qid'] for item in qid]
    data_path = "RAG-cy/wikidata/process/entity_rels"
    table_files = get_batch_files(data_path) 
    pool = Pool(processes = workers)
    value = 'qid'
    entity_rels = []
    num = 0
    for output, nums in tqdm(
            pool.imap_unordered(
                partial(filtering_relation, qid_values, value=value), table_files, chunksize=chunksize), 
            total=len(table_files)
        ):
            entity_rels.extend(output)
            num += nums
            if num == len(qid_values):
                break

    return entity_rels

#################################################################


#################### Get triples2 for entity #######################
def filtering_relation2(target_name, filename, value):
    filtered = []
    num = 0
    name  = ''
    for item in jsonl_generator(filename):
        if item[value] in target_name:
            filtered.append(item)
            if item[value] != name:
                num += 1
            name = item[value]
    return filtered, num


def get_entity_value(qid):
    qid_values = [item['qid'] for item in qid]
    data_path = "RAG-cy/wikidata/process/entity_values"
    table_files = get_batch_files(data_path) 
    pool = Pool(processes = workers)
    value = 'qid'
    entity_values = []
    num = 0
    for output, nums in tqdm(
            pool.imap_unordered(
                partial(filtering_relation2, qid_values, value=value), table_files, chunksize=chunksize), 
            total=len(table_files)
        ):
            entity_values.extend(output)
            num += nums
            if num == len(qid_values):
                break

    return entity_values
#################################################################


def Pruning_triples(triples_all, keys):
    threshold = 0.8
    triples_new = []

    k = []
    for item in keys:
        if 'tuples' in item:
            k.append(item['tuples'])
        elif 'triples' in item:
            k.append(item['triples'])

    for t in triples_all:
        t = f"({' ' if t['entity1'] is None else t['entity1']}, {' ' if t['rel'] is None else t['rel']}, {' ' if t['entity2'] is None else t['entity2']})"
        
        for key in k:
            try:
                # similarity = model.similarity(description, ques)
                similarity = calculate_similarity(t, key)
                # print(similarity)
                if similarity > threshold:
                    triples_new.append(t)
                    break
            except KeyError:
                continue
    
    return triples_new



##################### Get triples text ###########################################

def filtering_func2(target_name1, target_name2, filename, value):
    filtered = []
    for item in jsonl_generator(filename):
        if item[value] in target_name1 or item[value] in target_name2:
            filtered.append(item)
    return filtered


def filtering_rel_value(target_name1, target_name2, target_name3, filename, value):
    # entity_rel_value = {'entity1': None, 'rel': None, 'entity2': None}
    filtered = []
    for item in jsonl_generator(filename):
        if item[value] in target_name1 or item[value] in target_name2 or item[value] in target_name3:
            filtered.append(item)
    return filtered

def get_rel_text(entity_rels, entity_values):
    qid_values1 = [item['qid'] for item in entity_rels]
    qid_values2 = [item['value'] for item in entity_rels]
    pid = [item['property_id'] for item in entity_rels]

    qid_values3 = [item['qid'] for item in entity_values]
    pid2 = [item['property_id'] for item in entity_values]

    data_path1 = "RAG-cy/wikidata/process/labels"
    data_path2 = "RAG-cy/wikidata/process/plabels"
    table_files1 = get_batch_files(data_path1) 
    table_files2 = get_batch_files(data_path2) 
    pool = Pool(processes = workers)
    value1 = 'qid'
    value2 = 'pid'

    entity_value = []
    for output in tqdm(
            pool.imap_unordered(
                partial(filtering_rel_value, qid_values1, qid_values2, qid_values3, value=value1), table_files1, chunksize=chunksize), 
            total=len(table_files1)
        ):
            entity_value.extend(output)
    
    rel_value = []
    for output in tqdm(
            pool.imap_unordered(
                partial(filtering_func2, pid, pid2, value=value2), table_files2, chunksize=chunksize), 
            total=len(table_files1)
        ):
            rel_value.extend(output)

    result_rel = []


    for rel in entity_rels:
      
        rel_qid1 = rel["qid"]
        rel_qid2 = rel['value']
        rel_pid = rel['property_id']
      
        entity1 = None
        entity2 = None
        rel = None

        for entity in entity_value:
            if rel_qid1 != rel_qid2:
                if entity["qid"] == rel_qid1:
                    entity1 = entity["label"]
                    break  
                elif entity["qid"] == rel_qid2:
                    entity2 = entity["label"]
                    break
            else:
                if entity["qid"] == rel_qid1:
                    entity1 = entity["label"]
                    entity2 = entity["label"]
                    break
        
        for r in rel_value:
            if r['pid'] == rel_pid:
                rel = r['label']
                break
        result_rel.append({
            "entity1": entity1,
            "rel": rel,
            "entity2": entity2
        })

    result_pro = []
    for property in entity_values:
        entity2 = property['value']
        qid_p = property["qid"]
        pid_p = property["property_id"]

        entity_p = None
        rel_p = None

        for entity in entity_value:
            if entity["qid"] == qid_p:
                entity_p = entity["label"]
                break

        for r in rel_value:
            if r['pid'] == pid_p:
                rel_p = r['label']
                break
        
        result_pro.append({
            "entity1": entity_p,
            "rel": rel_p,
            "entity2": entity2
        })

    return result_rel, result_pro


#############################################################################



def replace_with_alias_key(triple, alias_to_key):
    # print(triple)
    triple = f"({' ' if triple['entity1'] is None else triple['entity1']}, {' ' if triple['rel'] is None else triple['rel']}, {' ' if triple['entity2'] is None else triple['entity2']})"
    parts = triple.split(', ')
    parts1 = parts[0].split('(')[1]
    parts2 = parts[-1].split(')')[0]
    
    if parts1 in alias_to_key:
        parts1 = alias_to_key[parts1]
   
    if parts2 in alias_to_key:
        parts2 = alias_to_key[parts2]

    result = '('+parts1+ ', '+ parts[1]+ ', ' + parts2+')'
    return result

def replace_with_key(target_str, alias_dict):
    # 遍历 alias_dict
    for key, aliases in alias_dict.items():
        if target_str in aliases:  # 如果目标字符串在别名列表中
            return key  # 返回对应的 key
    return target_str  # 如果没有找到匹配，返回原字符串

def unify(triples, triples_pro, alias):
    triples_all = triples + triples_pro
    result = defaultdict(list)

    # 遍历alias列表，将相同qid的alias放到同一个列表中
    for item in alias:
        result[item["entity"]].append(item["alias"])
    
    for key, value_list in result.items():
       
        merged_list = sum(value_list, [])
       
        result[key] = list(set(merged_list))

    alias_dict = dict(result)
  

    result = []
  
    for triple in triples_all:
    
        entity1 = triple['entity1']
        entity2 = triple['entity2']
        e1 = replace_with_key(entity1, alias_dict)
        e2 = replace_with_key(entity2, alias_dict)
        # if entity1 in result_dict
        triple['entity1'] = e1
        triple['entity2'] = e2
        result.append(triple)

    

    return result


def retrieval(question, keys): 
    keys = deduplication(keys)

    entity, relation = get_entity_relation(keys)
    print('entity:', entity)
    print('property:', relation)

    print('Get qid.......')
    qid, entity_and_alias = get_entity_id(entity)
    # pid = get_property_id(property)
    print("qid:", qid)
   

    print('Get description........')
    description = get_description(qid)
    # print('description:', description)
    
    print('Pruning description......')
    qid_new, description_new = Pruning_description(description, question)
    print('description:', description_new)
    description = [des['description'] for des in description_new]
   

    print('Get triples.............')
    qid = [item for item in qid if item['qid'] in qid_new]
    entity_rels = get_entity_rels(qid)
    entity_values = get_entity_value(qid)

    print('Get the texed triples...........')
    triples, triples_pro = get_rel_text(entity_rels, entity_values)
    

    # print('triples_text:', triples, len(triples))
    # print('triples_pro_text:', triples_pro, len(triples_pro))
    # print(entity_and_alias)
    print('Unify the triples!.........')
    triples = unify(triples, triples_pro, entity_and_alias)
    # print('get unified triples !')
    print(triples)
    print('Pruning triples......')
    triples_new = Pruning_triples(triples, keys)
    
  

    print('Retrival Done !')
    return list(set(triples_new)), description



if __name__ == "__main__":
   
    question = ['Were Scott Derrickson and Ed Wood of the same nationality?',
                'Who is Scott Derrickson?',
                'Who is Ed Wood?',
                'Are Scott Derrickson and Ed Wood of the same nationality?'
                ]
    keys = [
        {'entity': '(Claude Y Legault)'}, 
        {'entity': '(Ed Wood)'}, 
        {'entity': '(nationality)'}, 
        {'tuples': '(Scott Derrickson, nationality)'}, 
        {'tuples': '(Ed Wood, nationality)'}, 
        {'triples': '(Scott Derrickson, is, nationality)'}, 
        {'triples': '(Ed Wood, is, nationality)'}, 
        {'triples': '(Scott Derrickson, and, Ed Wood)'}, 
        {'triples': '(Scott Derrickson, of the same nationality, Ed Wood)'}]
    
    keys = deduplication(keys)

    entity, relation = get_entity_relation(keys)
    print('entity:', entity)
    print('property:', relation)

    print('Get qid.......')
    qid, entity_and_alias = get_entity_id(entity)
    # pid = get_property_id(property)
    print("qid:", qid)
   
    print('Get description........')
    description = get_description(qid)
    # print('description:', description)
    
    print('Pruning description......')
    qid_new, description_new = Pruning_description(description, question)
    print('description:', description_new)
    description = [des['description'] for des in description_new]
    # print(description)

    print('Get triples.............')
    qid = [item for item in qid if item['qid'] in qid_new]
    entity_rels = get_entity_rels(qid)
    entity_values = get_entity_value(qid)

    print('Get the texed triples...........')
    triples, triples_pro = get_rel_text(entity_rels, entity_values)
    

    # print('triples_text:', triples, len(triples))
    # print('triples_pro_text:', triples_pro, len(triples_pro))
    # print(entity_and_alias)
    print('Unify the triples!.........')
    triples = unify(triples, triples_pro, entity_and_alias)
    # print('get unified triples !')

    print('Pruning triples......')
    triples_new = Pruning_triples(triples, keys)
    
    print('new_triples:', triples_new, len(triples_new))
    # print('qid_new, len_qid:', qid, len(qid))
    # print('description_new, len_description:', description_new, len(description_new))

    print('Retrival Done !')
    
   