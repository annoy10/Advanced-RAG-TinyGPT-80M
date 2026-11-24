from guardrails.faithfulness_checker import check_faithfulness
def test_faithfulness_flags_unrelated():
    assert check_faithfulness("Zebras live in the moon.", "A report about football.")["faithful"] is False
