from llm_serving.quality import parse_lm_eval_results


def test_parser_prefers_record_f1_over_sample_length_metadata():
    results = parse_lm_eval_results(
        {
            "results": {
                "record": {
                    "sample_len": 10,
                    "f1,none": 0.8,
                    "f1_stderr,none": 0.13,
                    "em,none": 0.7,
                }
            }
        }
    )

    assert len(results) == 1
    assert results[0].primary_metric_name == "f1,none"
    assert results[0].primary_score == 0.8
    assert results[0].stderr == 0.13
