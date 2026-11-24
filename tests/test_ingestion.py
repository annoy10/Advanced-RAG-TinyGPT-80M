from ingestion.cleaning import clean_text

def test_cleaning():
    assert clean_text(" a   b \n\n\n c ") == "a b \n\n c"
