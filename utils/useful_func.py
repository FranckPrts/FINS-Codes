
def remove_double(rej2:list, rej3:list):
    merged_list=rej3
    for i in rej2:
        if i not in rej3:
            merged_list.append(i)
    return merged_list