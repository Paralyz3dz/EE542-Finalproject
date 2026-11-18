import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dotenv import load_dotenv

from src.db.connection import ROOT
from src.rag import retriever
from src.rag.corpus_hygiene import load_policy
from src.workflow.engine import load_scenario
from eval.run_candidate_experiment import corpus_state, assert_fixed_inputs, digest, write_json, source_metrics, compact


ORDER = ((False, True), (True, False), (False, True))


def observe(question, vector, enabled, exclusions):
    trace = {}
    original_vector = retriever.vector_search
    original_keyword = retriever.bm25_search

    def dense(*args, **kwargs):
        ids = set(kwargs.get('excluded_ids', []))
        if ids != (exclusions if enabled else set()):
            raise ValueError('Unexpected vector exclusions')
        rows = original_vector(*args, **kwargs)
        trace['vector'] = [{'chunk_id':r[0], 'citation':r[1], 'score':float(r[3])} for r in rows]
        return rows

    def sparse(*args, **kwargs):
        trace['eligible_chunks'] = len(args[1])
        if enabled and {c['id'] for c in args[1]} & exclusions:
            raise ValueError('Noise entered the keyword index')
        result = original_keyword(*args, **kwargs)
        trace['bm25'] = compact(result)
        return result

    with patch.object(retriever, 'embed_query', return_value=vector), patch.object(retriever, 'vector_search', side_effect=dense), patch.object(retriever, 'bm25_search', side_effect=sparse):
        start = time.perf_counter()
        output = retriever.retrieve(question['retrieval_query'], candidate_k=10, exclude_reference_chunks=enabled)
        elapsed = time.perf_counter() - start
    matches = output['matches']
    route_ids = {r['chunk_id'] for r in trace['vector'] + trace['bm25'] + matches}
    if enabled and route_ids & exclusions:
        raise ValueError('Excluded noise entered retrieval results')
    if len(matches) > 5:
        raise ValueError('Output limit changed')
    noisy = [m['chunk_id'] for m in matches[:3] if m['chunk_id'] in exclusions]
    return {'id':question['id'], 'query':question['retrieval_query'], 'source':question['source'],
            'cleanup':enabled, 'candidate_k':10, 'elapsed_seconds':elapsed,
            **source_metrics(question,matches), 'noise_in_top3':noisy,
            'routes':trace, 'matches':compact(matches)}, matches


def summarize(rows, questions):
    expected = {(n,flag,q['id']) for n in (1,2,3) for flag in (False,True) for q in questions}
    if len(rows) != len(expected) or {(r['round'],r['cleanup'],r['id']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate runs')
    result = {}
    for flag in (False,True):
        group = [r for r in rows if r['cleanup']==flag]
        result['cleaned' if flag else 'baseline'] = {
            'hit_at_3':float(np.mean([r['hit_at_3'] for r in group])),
            'hit_at_5':float(np.mean([r['hit_at_5'] for r in group])),
            'mrr_at_5':float(np.mean([r['mrr_at_5'] for r in group])),
            'noise_affected_queries':sum(bool(r['noise_in_top3']) for r in group),
            'noise_passages_in_top3':sum(len(r['noise_in_top3']) for r in group),
            'short_vector_candidate_lists':sum(len(r['routes']['vector'])<10 for r in group),
            'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in group])),
            'p95_seconds':float(np.percentile([r['elapsed_seconds'] for r in group],95)),
            'rounds':[{'round':n,'hits_at_3':sum(r['hit_at_3'] for r in group if r['round']==n),
                       'hits_at_5':sum(r['hit_at_5'] for r in group if r['round']==n),
                       'noise_queries':sum(bool(r['noise_in_top3']) for r in group if r['round']==n)} for n in (1,2,3)]}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--input',type=Path,default=ROOT/'eval/results/candidate-pool-2026-11-17')
    parser.add_argument('--out',type=Path,default=ROOT/'eval/results/corpus-hygiene-2026-11-17')
    parser.add_argument('--query-cache',type=Path,default=ROOT/'data/processed/candidate_queries.json')
    args=parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    failure=args.out/'incomplete.json'
    if failure.exists() and json.loads(failure.read_text()).get('invalidated'):
        raise ValueError('Invalidated run; choose a new output directory')
    previous=json.loads((args.input/'manifest.json').read_text(encoding='utf-8'))
    questions=json.loads((args.input/'questions.json').read_text(encoding='utf-8'))
    _,live_questions=load_scenario('intake')
    if len(questions)!=104 or digest(questions)!=digest(live_questions):
        raise ValueError('Question set changed')
    cache=json.loads(args.query_cache.read_text(encoding='utf-8'))
    identity={'questions':digest(questions),'model':previous['model'],'instruction':retriever.QUERY_INSTRUCTION,'dimensions':1024}
    vectors=np.asarray(cache['vectors'],dtype=np.float32)
    if cache['identity']!=identity or vectors.shape!=(104,1024) or not np.isfinite(vectors).all() or not np.all(np.any(vectors,axis=1)):
        raise ValueError('Invalid or stale query vectors')
    if hashlib.sha256(vectors.tobytes()).hexdigest()!=previous['query_vectors_sha256']:
        raise ValueError('Query vectors differ from the previous baseline')
    if os.getenv('EMBEDDING_MODEL','Qwen/Qwen3-Embedding-0.6B')!=previous['model']['embedding_model']:
        raise ValueError('Embedding model changed')
    corpus=corpus_state()
    if digest(corpus)!=digest(previous['corpus']):
        raise ValueError('The physical baseline corpus changed')
    settings=json.loads((ROOT/'config/settings.json').read_text())
    if (settings['tax_year'],settings['retrieve_top_k'],settings['retrieve_candidate_k'],settings['rrf_constant'])!=(2025,5,10,60):
        raise ValueError('Keep the selected K=10 settings')
    policy=load_policy()
    excluded={e['id'] for e in policy['exclusions']}
    files=('src/rag/retriever.py','src/rag/hybrid.py','src/rag/corpus_hygiene.py','src/db/queries.py',
           'eval/run_candidate_experiment.py','eval/run_hygiene_experiment.py','config/retrieval_exclusions.json')
    manifest={'questions_sha256':digest(questions),'settings':settings,'corpus':corpus,'model':previous['model'],
              'query_vectors_sha256':previous['query_vectors_sha256'],'exclusions':len(excluded),'order':ORDER,
              'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'change':'Only the reviewed index/reference-block eligibility policy. K=10, output=5, display=3, embeddings and RRF=60 stay fixed.',
              'timing':'Full production retrieval excluding query embedding, answer generation and judging.'}
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):
        raise ValueError('Run fingerprint changed; choose a new output directory')
    write_json(path,manifest)
    write_json(args.out/'questions.json',questions)
    path=args.out/'results.jsonl'
    records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    done={(r['round'],r['cleanup'],r['id']) for r in records}
    passage_path=args.out/'passages.json'
    passages=json.loads(passage_path.read_text(encoding='utf-8')) if passage_path.exists() else {}
    try:
        for n,flags in enumerate(ORDER,1):
            for flag in flags:
                assert_fixed_inputs(manifest)
                for q,v in zip(questions[:3],vectors[:3]):
                    observe(q,v,flag,excluded)
                with path.open('a',encoding='utf-8',newline='\n') as output:
                    for i,(q,v) in enumerate(zip(questions,vectors),1):
                        key=(n,flag,q['id'])
                        if key in done:
                            continue
                        row,matches=observe(q,v,flag,excluded)
                        row['round']=n
                        for m in matches:
                            passages[m['chunk_id']]={'citation':m['citation'],'text':m['text']}
                        write_json(passage_path,passages)
                        output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
                        records.append(row);done.add(key)
                        if i%20==0 or i==104:
                            print(f'Round {n} cleanup={flag}: {i}/104',flush=True)
                assert_fixed_inputs(manifest)
        summary=summarize(records,questions)
        prior=[json.loads(s) for s in (args.input/'results.jsonl').read_text(encoding='utf-8').splitlines()]
        old={r['id']:r for r in prior if r['round']==1 and r['candidate_k']==10}
        summary['baseline_reproduction']={str(n):[r['id'] for r in records if r['round']==n and not r['cleanup'] and
             (r['hit_at_3'],r['hit_at_5'])!=(old[r['id']]['hit_at_3'],old[r['id']]['hit_at_5'])] for n in (1,2,3)}
        lookup={(r['round'],r['cleanup'],r['id']):r for r in records}
        changes=[]
        for n in (1,2,3):
            for q in questions:
                before,after=lookup[n,False,q['id']],lookup[n,True,q['id']]
                if before['hit_at_3']!=after['hit_at_3'] or before['noise_in_top3']:
                    changes.append({'round':n,'id':q['id'],'source':q['source'],'before':before,'after':after})
        write_json(args.out/'changes.json',changes)
        assert_fixed_inputs(manifest)
        write_json(args.out/'corpus_after.json',corpus)
        write_json(args.out/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        message=re.sub(r'postgres(?:ql)?://\S+|sk-[\w-]+','[REDACTED]',str(exc))
        write_json(failure,{'error':type(exc).__name__+': '+message[:1000],'invalidated':isinstance(exc,ValueError)})
        raise


if __name__=='__main__':
    main()
