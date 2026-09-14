#!/usr/bin/env python3
"""Deterministic CPU ranking; real selection freezing remains Stage4-gated."""
import argparse,json
from pathlib import Path
from checkpoint_selection import ranking_key,read

def rank_candidates(summaries,candidates):
    expected={c['checkpoint_id']:c for c in candidates}
    if len(summaries)!=len(expected) or {s['checkpoint_id'] for s in summaries}!=set(expected):raise ValueError('Incomplete candidate set')
    for s in summaries:
        ranking_key(s)
        if s['training_groups']!=expected[s['checkpoint_id']]['training_groups']:raise ValueError('Checkpoint budget mismatch')
    ranking=sorted(summaries,key=ranking_key);keys=['accuracy','unparseable','truncation','earlier_checkpoint','checkpoint_id'];trace=[]
    for row in ranking[1:]:
        a,b=ranking_key(ranking[0]),ranking_key(row)
        trace.append(dict(winner=ranking[0]['checkpoint_id'],other=row['checkpoint_id'],deciding_key=next(keys[i] for i in range(5) if a[i]!=b[i]),winner_key=list(a),other_key=list(b)))
    return dict(selected_checkpoint=ranking[0]['checkpoint_id'],full_ranking=ranking,tie_break_trace=trace)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summaries',required=True);p.add_argument('--candidates',required=True);a=p.parse_args()
    print(json.dumps(rank_candidates(read(a.summaries),read(a.candidates)),ensure_ascii=False,indent=2))
