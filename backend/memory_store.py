memory_dict = {}

def store_memory(query, answer):
    memory_dict[query] = answer

def recall_memory(query):
    return memory_dict.get(query)

