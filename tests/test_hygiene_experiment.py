import unittest

from eval.run_hygiene_experiment import summarize


class HygieneExperimentTests(unittest.TestCase):
    def records(self):
        questions = [{"id":str(i)} for i in range(104)]
        rows = []
        for n in (1,2,3):
            for enabled in (False,True):
                for i in range(104):
                    rows.append({"round":n,"cleanup":enabled,"id":str(i),
                                 "hit_at_3":i<(75 if enabled else 71),
                                 "hit_at_5":i<(84 if enabled else 81),"mrr_at_5":0.5,
                                 "noise_in_top3":[] if enabled or i>=15 else ["index"],
                                 "elapsed_seconds":1.0,"routes":{"vector":[{}]*10}})
        return rows,questions

    def test_summary_separates_source_metrics_noise_and_repetitions(self):
        rows,questions=self.records()
        result=summarize(rows,questions)
        self.assertEqual(result['baseline']['rounds'][0]['hits_at_3'],71)
        self.assertEqual(result['cleaned']['rounds'][0]['hits_at_3'],75)
        self.assertEqual(result['baseline']['noise_affected_queries'],45)
        self.assertEqual(result['cleaned']['noise_affected_queries'],0)
        self.assertEqual(result['cleaned']['short_vector_candidate_lists'],0)

    def test_partial_and_duplicate_runs_are_not_reported_as_complete(self):
        rows,questions=self.records()
        for partial in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):
                summarize(partial,questions)


if __name__=='__main__':
    unittest.main()
