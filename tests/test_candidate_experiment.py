import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from eval.run_candidate_experiment import source_metrics, summarize, assert_fixed_inputs, cache_vectors


class CandidateExperimentTests(unittest.TestCase):
    def test_source_metrics_use_fixed_top_three_and_top_five(self):
        question = {"source": "IRS Pub 17; IRS Pub 505"}
        matches = [{"citation": "Other (2025), p. 1"}] * 4 + [{"citation": "irs pub 505 (2025), p. 3"}]
        result = source_metrics(question, matches)
        self.assertFalse(result["hit_at_3"])
        self.assertTrue(result["hit_at_5"])
        self.assertEqual(result["mrr_at_5"], 0.2)
        self.assertFalse(source_metrics(question, matches[:4])["hit_at_5"])

    def test_changed_corpus_invalidates_the_run(self):
        with patch('eval.run_candidate_experiment.corpus_state', return_value={'sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'invalidate'):
                assert_fixed_inputs({'corpus': {'sha256': 'original'}})

    def records(self):
        questions = [{"id": str(i)} for i in range(104)]
        rows = []
        for n in range(1, 4):
            for k, hits3, hits5 in ((5, 70, 80), (10, 72, 81), (20, 73, 79)):
                for i in range(104):
                    rows.append({"round": n, "candidate_k": k, "id": str(i), "hit_at_3": i < hits3,
                                 "hit_at_5": i < hits5, "mrr_at_5": float(i < hits5),
                                 "candidate_source_coverage": True, "elapsed_seconds": k / 10})
        return questions, rows

    def test_hit_at_five_regression_disqualifies_larger_hit_at_three_gain(self):
        questions, rows = self.records()
        result = summarize(rows, questions)
        self.assertEqual(result["selected_candidate_k"], 10)
        self.assertFalse(result["acceptance_passed"])
        self.assertEqual(result["conditions"]["5"]["rounds"][0]["hits_at_3"], 70)

    def test_no_improvement_retains_baseline(self):
        questions, rows = self.records()
        for row in rows:
            row['hit_at_3'] = int(row['id']) < 70
            row['hit_at_5'] = int(row['id']) < 80
        self.assertEqual(summarize(rows, questions)['selected_candidate_k'], 5)

    def test_quality_ties_use_latency_then_smaller_pool(self):
        questions, rows = self.records()
        for row in rows:
            if row['candidate_k'] != 5:
                row['hit_at_3'] = int(row['id']) < 72
                row['hit_at_5'] = int(row['id']) < 81
                row['elapsed_seconds'] = 1.0 if row['candidate_k'] == 20 else 2.0
        self.assertEqual(summarize(rows, questions)['selected_candidate_k'], 20)
        for row in rows:
            if row['candidate_k'] != 5:
                row['elapsed_seconds'] = 1.0
        self.assertEqual(summarize(rows, questions)['selected_candidate_k'], 10)

    def test_stale_query_cache_is_not_reused_or_silently_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queries.json'
            path.write_text(json.dumps({'identity': {}, 'vectors': []}))
            with patch('eval.run_candidate_experiment.retriever.embed_texts') as embed:
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    cache_vectors(path, [{'id': 'changed-query'}], {'embedding_model': 'model'})
                embed.assert_not_called()

    def test_partial_and_duplicate_runs_are_rejected(self):
        questions, rows = self.records()
        for incomplete in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(incomplete, questions)


if __name__ == '__main__':
    unittest.main()
