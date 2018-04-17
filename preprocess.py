#!/usr/bin/python3
from tqdm import tqdm

import json
import sys

N_NEIGHBORS = 3
PAD_STR = "0PAD"
VAR_PREFIX = "MLPL_"
MY_VAR = "0VAR"
OTHER_VAR = "1VAR"
SEP_SYM = "0MID"
IGNORED_TOKENS = {"{", "}", "(", ")", ";"}

total = 0
success = 0

def process_file(in_file, out_file):
    global total
    global success
    with open(out_file, "w") as fout:
        with open(in_file, "r") as fin:
            for line in tqdm(fin):
                idx = line.find(" ")
                script = line[:idx]
                # print("Processing "+script+ " ("+ str(success) + ", " + str(total-success) + ")")
                total += 1
                json_ast_str = line[idx+1:]
                try:
                    json_ast = json.loads(json_ast_str)
                    map = dict()
                    process_json(json_ast, [], map)
                    for (k, v) in map.items():
                        k = k[k.find('_')+1:]
                        id = k[0:k.find('_')]
                        k = k[k.find('_')+1:]
                        sample = {'script': script, 'id': id, 'input': v, 'output': k}
                        fout.write(json.dumps(sample))
                        fout.write('\n')
                    success += 1
                except Exception as e:
                    print(e)


def process_json(json_ast, stack, map):
    for i, e in enumerate(json_ast):
        if isinstance(e, list):
            stack.append((json_ast, i))
            process_json(e, stack, map)
            stack.pop()
        elif e.startswith(VAR_PREFIX):
            context = get_context(stack, e)
            if not (e in map):
                map[e] = list()
            map[e].append(context)



def get_context(stack, this_var):
    before = []
    after = []
    for (tree, i) in reversed(stack):
        (pre, post) = collect_neighbors(tree, i, this_var)
        done = False
        for e in post:
            after.append(e)
            if len(after) >= N_NEIGHBORS:
                done = True
                break
        for e in reversed(pre):
            before.append(e)
            if len(before) >= N_NEIGHBORS:
                done = True
                break
        if done:
            break
    for j in range(len(before), N_NEIGHBORS):
        before.append(PAD_STR)
    for j in range(len(after), N_NEIGHBORS):
        after.append(PAD_STR)
    before.reverse()
    before.append(SEP_SYM)
    before.extend(after)
    return before


def collect_neighbors(json_ast, i, this_var):
    pre = []
    post = []
    node_label = ""
    for e in json_ast:
        if not isinstance(e, list):
            if node_label != "":
                node_label += "@"
            node_label += e
    if node_label != "":
        post.append(node_label)
    for j in range(0, i):
        if isinstance(json_ast[j], list):
           serialize(json_ast[j], pre, this_var, True)
    for j in range(i+1, len(json_ast)):
        if isinstance(json_ast[j], list):
           serialize(json_ast[j], post, this_var, False)
    return (pre, post)

def serialize(json_ast, arr, this_var, is_pre):
    node_label = ""
    idx = len(arr)
    for e in json_ast:
        if not isinstance(e, list):
            if node_label != "":
                node_label += "@"
            if e == this_var:
                node_label += MY_VAR
            elif False and e.startswith(VAR_PREFIX):
                node_label += OTHER_VAR
            else:
                node_label += e
    if node_label != "":
        if not is_pre:
            arr.append(node_label)
    for e in json_ast:
        if isinstance(e, list):
            serialize(e, arr, this_var, is_pre)
    if node_label != "":
        if is_pre:
            arr.append(node_label)


process_file(sys.argv[1], sys.argv[2])
