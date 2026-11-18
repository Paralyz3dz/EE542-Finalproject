import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from dotenv import load_dotenv

from src.db.connection import ROOT, get_conn
from src.rag import retriever
from src.workflow.engine import _doc_key, load_scenario


ORDER = ((5, 10, 20), (10, 20, 5), (20, 5, 10))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def corpus_state():
    with get_conn() as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        rows = conn.execute("""SELECT id, tax_year, source, content_hash, embedding_model,
            md5(text), md5(embed_text), md5(citation), md5(embedding::text), type,
            printed_pages, pdf_pages FROM chunks WHERE tax_year=2025 ORDER BY id""").fetchall()
        indexes = conn.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='chunks' ORDER BY indexname").fetchall()
        search = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    if len(rows) != 6004 or len({r[2] for r in rows}) != 21 or any(r[8] is None for r in rows):
        raise ValueError('Expected the frozen 21-source, 6004-vector corpus')
    return {'chunks': len(rows), 'documents': 21, 'sha256': digest(rows), 'indexes': indexes, 'hnsw_ef_search': search}


def assert_fixed_inputs(manifest):
    if corpus_state() != manifest['corpus']:
        raise ValueError('Corpus changed; invalidate this run')
    settings = json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))
    _, questions = load_scenario('intake')
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in manifest['code_sha256']}
    if settings != manifest['settings'] or digest(questions) != manifest['questions_sha256'] or code != manifest['code_sha256']:
        raise ValueError('Experiment inputs changed; invalidate this run')


def source_metrics(question, matches):
    sources = {_doc_key(s) for s in question['source'].split(';') if s.strip()}
    ranks = [i + 1 for i, m in enumerate(matches[:5]) if _doc_key(m['citation']) in sources]
    return {'hit_at_3': bool(ranks and ranks[0] <= 3), 'hit_at_5': bool(ranks),
            'mrr_at_5': 1 / ranks[0] if ranks else 0.0}


def compact(matches):
    keys = ('chunk_id', 'citation', 'score', 'vector_score', 'bm25_score', 'vector_rank', 'bm25_rank')
    return [{k: m[k] for k in keys if k in m} for m in matches]


def observe(question, vector, count):
    trace = {}
    original_vector = retriever.vector_search
    original_keyword = retriever.bm25_search

    def dense(*args, **kwargs):
        rows = original_vector(*args, **kwargs)
        trace['vector'] = [{'chunk_id': r[0], 'citation': r[1], 'score': float(r[3])} for r in rows]
        return rows

    def sparse(*args, **kwargs):
        matches = original_keyword(*args, **kwargs)
        trace['bm25'] = compact(matches)
        return matches

    with patch.object(retriever, 'embed_query', return_value=vector), patch.object(retriever, 'vector_search', side_effect=dense), patch.object(retriever, 'bm25_search', side_effect=sparse):
        started = time.perf_counter()
        result = retriever.retrieve(question['retrieval_query'], candidate_k=count)
        elapsed = time.perf_counter() - started
    matches = result['matches']
    if len(matches) > 5 or len(trace['vector']) > count or len(trace['bm25']) > count:
        raise ValueError('Unexpected candidate or output count')
    metrics = source_metrics(question, matches)
    metrics['candidate_source_coverage'] = bool(
        {_doc_key(s) for s in question['source'].split(';')} &
        {_doc_key(m['citation']) for m in trace['vector'] + trace['bm25']})
    metrics['top3_source_count'] = len({_doc_key(m['citation']) for m in matches[:3]})
    return {'id': question['id'], 'query': question['retrieval_query'], 'expected_source': question['source'],
            'candidate_k': count, 'elapsed_seconds': elapsed, **metrics,
            'routes': trace, 'matches': compact(matches)}, matches


def summarize(records, questions):
    expected = {(r, k, q['id']) for r in range(1, 4) for k in (5, 10, 20) for q in questions}
    actual = {(row['round'], row['candidate_k'], row['id']) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError('Incomplete or duplicated experiment records')
    summary = {}
    for count in (5, 10, 20):
        rows = [r for r in records if r['candidate_k'] == count]
        times = [r['elapsed_seconds'] for r in rows]
        summary[str(count)] = {
            'hit_at_3': float(np.mean([r['hit_at_3'] for r in rows])),
            'hit_at_5': float(np.mean([r['hit_at_5'] for r in rows])),
            'mrr_at_5': float(np.mean([r['mrr_at_5'] for r in rows])),
            'candidate_source_coverage': float(np.mean([r['candidate_source_coverage'] for r in rows])),
            'mean_seconds': float(np.mean(times)), 'p95_seconds': float(np.percentile(times, 95)),
            'rounds': [{'round': n, 'hits_at_3': sum(r['hit_at_3'] for r in rows if r['round'] == n),
                        'hits_at_5': sum(r['hit_at_5'] for r in rows if r['round'] == n)} for n in range(1, 4)]}
    baseline = summary['5']
    eligible = [k for k in (10, 20) if summary[str(k)]['hit_at_3'] > baseline['hit_at_3'] and summary[str(k)]['hit_at_5'] >= baseline['hit_at_5']]
    winner = min(eligible, key=lambda k: (-summary[str(k)]['hit_at_3'], -summary[str(k)]['hit_at_5'], summary[str(k)]['p95_seconds'], k)) if eligible else 5
    return {'conditions': summary, 'selected_candidate_k': winner,
            'acceptance_passed': summary[str(winner)]['hit_at_3'] >= 84 / 104,
            'historical_baseline_hits': 68,
            'historical_baseline_difference': [r['hits_at_3'] - 68 for r in baseline['rounds']]}


def cache_vectors(path, questions, model):
    identity = {'questions': digest(questions), 'model': model,
                'instruction': retriever.QUERY_INSTRUCTION, 'dimensions': retriever.EMBEDDING_DIM}
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['identity'] != identity:
            raise ValueError('Query cache fingerprint differs; use a new cache path')
        vectors = np.asarray(data['vectors'], dtype=np.float32)
    else:
        vectors = []
        for i in range(0, len(questions), 8):
            batch = questions[i:i + 8]
            vectors.extend(retriever.embed_texts([f"Instruct: {retriever.QUERY_INSTRUCTION}\nQuery: {q['retrieval_query'].strip()}" for q in batch]).tolist())
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {'identity': identity, 'vectors': vectors})
        vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (104, 1024) or not np.isfinite(vectors).all() or not np.all(np.any(vectors, axis=1)):
        raise ValueError('Invalid query vectors')
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'eval/results/candidate-pool-2026-11-17')
    parser.add_argument('--query-cache', type=Path, default=ROOT / 'data/processed/candidate_queries.json')
    args = parser.parse_args()
    invalid = args.out / 'incomplete.json'
    if invalid.exists() and json.loads(invalid.read_text(encoding='utf-8')).get('invalidated'):
        raise ValueError('This experiment was invalidated; choose a new output directory')
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    settings = json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))
    if settings['tax_year'] != 2025 or settings['retrieve_top_k'] != 5 or settings['rrf_constant'] != 60:
        raise ValueError('Unexpected experiment settings')
    _, questions = load_scenario('intake')
    if len(questions) != 104 or len({q['id'] for q in questions}) != 104:
        raise ValueError('Expected 104 unique interview questions')
    model_name = os.getenv('OLLAMA_EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    with urlopen(base + '/api/tags', timeout=15) as response:
        tags = json.load(response)['models']
    tag = next(t for t in tags if t['name'] == model_name)
    model = {'embedding_model': os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
             'ollama_model': model_name, 'digest': tag['digest']}
    if model['embedding_model'] != 'Qwen/Qwen3-Embedding-0.6B' or model_name != 'qwen3-embedding:0.6b':
        raise ValueError('Keep the baseline embedding model')
    print('Preparing fingerprinted query vectors for 104 questions.', flush=True)
    vectors = cache_vectors(args.query_cache, questions, model)
    corpus = corpus_state()
    print('Frozen 6004 chunks; starting nine full retrieval runs.', flush=True)
    files = ('src/rag/retriever.py', 'src/rag/hybrid.py', 'src/db/queries.py', 'eval/run_candidate_experiment.py')
    manifest = {'questions_sha256': digest(questions), 'settings': settings, 'model': model,
                'query_vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(), 'corpus': corpus,
                'code_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
                'condition_order': ORDER, 'repetitions': 3, 'warmup_queries_per_condition': 3,
                'timing': 'Full production retrieval, including corpus fetch and BM25 construction; excludes embedding, answer generation and LLM judging.',
                'metric': 'Source-name coverage only; passage relevance and answer correctness are not scored.'}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != json.loads(json.dumps(manifest)):
        raise ValueError('Experiment fingerprint differs; choose a new output directory')
    write_json(manifest_path, manifest)
    write_json(args.out / 'questions.json', questions)
    record_path = args.out / 'results.jsonl'
    records = [json.loads(line) for line in record_path.read_text(encoding='utf-8').splitlines()] if record_path.exists() else []
    done = {(r['round'], r['candidate_k'], r['id']) for r in records}
    passage_path = args.out / 'passages.json'
    passages = json.loads(passage_path.read_text(encoding='utf-8')) if passage_path.exists() else {}
    try:
        for round_no, counts in enumerate(ORDER, 1):
            for count in counts:
                assert_fixed_inputs(manifest)
                for question, vector in zip(questions[:3], vectors[:3]):
                    observe(question, vector, count)
                with record_path.open('a', encoding='utf-8', newline='\n') as output:
                    for i, (question, vector) in enumerate(zip(questions, vectors), 1):
                        key = (round_no, count, question['id'])
                        if key in done:
                            continue
                        row, matches = observe(question, vector, count)
                        row['round'] = round_no
                        for match in matches:
                            passages[match['chunk_id']] = {'citation': match['citation'], 'text': match['text']}
                        write_json(passage_path, passages)
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                        records.append(row)
                        done.add(key)
                        if i % 20 == 0 or i == len(questions):
                            print(f'Round {round_no} K={count}: {i}/104', flush=True)
                assert_fixed_inputs(manifest)
        summary = summarize(records, questions)
        changes = []
        for count in (10, 20):
            for q in questions:
                paired = []
                for n in range(1, 4):
                    before = next(r for r in records if (r['round'], r['candidate_k'], r['id']) == (n, 5, q['id']))
                    after = next(r for r in records if (r['round'], r['candidate_k'], r['id']) == (n, count, q['id']))
                    if before['hit_at_3'] != after['hit_at_3']:
                        paired.append({'round': n, 'change': 'gain' if after['hit_at_3'] else 'loss',
                                       'baseline': before['matches'], 'experiment': after['matches']})
                if paired:
                    changes.append({'id': q['id'], 'query': q['retrieval_query'], 'source': q['source'], 'candidate_k': count, 'rounds': paired})
        write_json(args.out / 'changes.json', changes)
        assert_fixed_inputs(manifest)
        write_json(args.out / 'corpus_after.json', corpus)
        write_json(args.out / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        message = re.sub(r'postgres(?:ql)?://\S+|sk-[\w-]+', '[REDACTED]', str(exc))
        write_json(args.out / 'incomplete.json', {'error': type(exc).__name__ + ': ' + message[:1000], 'invalidated': isinstance(exc, ValueError)})
        raise


if __name__ == '__main__':
    main()
