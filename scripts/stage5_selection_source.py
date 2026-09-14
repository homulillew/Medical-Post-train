"""Preserve the frozen selection source, including its single missing-C option label."""
import csv
import re
from medical_posttrain.data.exam import messages, options_schema, canonical_answer

GAP_ID='cmexam_val:fadb22c89beb1b7115dc36460ba792eb96b7b972:6027'
GAP_OPTIONS='A 抑制二氢叶酸合成酶\nB 改变细菌细胞膜通透性\nD 抑制二氢叶酸还原酶\nE 抑制细菌DNA螺旋酶'


def source_options(pid,row):
    try:return options_schema(row['Options'])
    except ValueError as exc:
        # Only the documented source record is compatible; unexpected schema changes fail closed.
        if str(exc)!='noncontiguous_options' or pid!=GAP_ID or row['Options']!=GAP_OPTIONS or row['Answer']!='A':raise
        options={line[0]:line[2:] for line in row['Options'].splitlines()}
        assert list(options)==['A','B','D','E']
        return options


def selection_inputs(protocol):
    with open(protocol['dataset']['source_metadata']['path']) as f:source=list(csv.DictReader(f))
    items=[];requests=[]
    for pid in protocol['dataset']['prompt_ids']:
        row=source[int(pid.rsplit(':',1)[1])];options=source_options(pid,row)
        answer=canonical_answer(row['Answer'],''.join(options))
        assert answer is not None
        item=dict(id=pid,question=row['Question'],options=options,answer=answer)
        items.append(item);requests.append(dict(id=pid,messages=messages(dict(item,answer_set=answer))))
    return items,requests
