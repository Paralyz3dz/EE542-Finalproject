from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from eval.run_baseline import (
    faithfulness_score, judge_payload, retrieval_metrics,
    RetrievalObserver, run_agent_case,
)


class MetricTests(unittest.TestCase):
    def test_recall_and_mrr_measure_different_things(self):
        scores = retrieval_metrics(["noise", "a", "other"], ["a", "b"])
        self.assertEqual(scores["recall_at_5"], 0.5)
        self.assertEqual(scores["mrr_at_5"], 0.5)
        self.assertEqual(retrieval_metrics(["a"], ["a", "b"])["mrr_at_5"], 1)

    def test_misses_and_results_beyond_five_score_zero(self):
        for ids in ([], ["x", "y", "z", "u", "v", "a"]):
            self.assertEqual(retrieval_metrics(ids, ["a"]), {"recall_at_5": 0.0, "mrr_at_5": 0.0})

    def test_unsupported_claims_lower_faithfulness_and_no_claims_are_undefined(self):
        self.assertEqual(faithfulness_score({"claims": [{"supported": True}, {"supported": False}]}), 0.5)
        self.assertIsNone(faithfulness_score({"claims": []}))

    def test_faithfulness_judge_never_receives_gold_answer_or_gold_context(self):
        sample = {"question": "Question", "standard_answer": "Gold answer", "relevant_documents": [{"text": "Gold passage"}]}
        response = {"answer": "Generated answer", "retrieval_calls": [{"rrf": [{"chunk_id": "a", "citation": "Actual citation", "text": "Actual retrieved passage"}]}]}
        payload = judge_payload(sample, response, "faithfulness")
        self.assertNotIn("reference", payload)
        self.assertEqual(payload["retrieved_contexts"], [{"citation": "Actual citation", "text": "Actual retrieved passage"}])
        self.assertEqual(judge_payload(sample, response, "correctness")["reference"], "Gold answer")

    def test_generation_receives_only_original_user_question(self):
        model = MagicMock()
        model.invoke.return_value = {"messages": [AIMessage(content="An answer")]}
        sample = {"id": "sample", "question": "Question", "standard_answer": "Secret gold answer"}
        result = run_agent_case(model, RetrievalObserver(), sample)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(model.invoke.call_args.args[0], {"messages": [{"role": "user", "content": "Question"}]})

    def test_observer_keeps_parallel_case_contexts_separate(self):
        observer = RetrievalObserver()
        def fake_retrieve(query):
            observer.current_call.get()["rrf"] = [{"chunk_id": query}]
            return {"matches": [{"chunk_id": query}], "tax_year": 2025}
        observer.original_retrieve = fake_retrieve
        def run(query):
            calls = []
            token = observer.current_case.set(calls)
            try:
                observer.observe_retrieve(query)
            finally:
                observer.current_case.reset(token)
            return calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ["first", "second"]))
        self.assertEqual(results[0][0]["rrf"][0]["chunk_id"], "first")
        self.assertEqual(results[1][0]["rrf"][0]["chunk_id"], "second")


if __name__ == "__main__":
    unittest.main()
