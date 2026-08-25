from amp_finder.data import FastaRecord, clean_positive_records, sample_negative_fragments


def test_positive_cleaning_deduplicates_and_reports_invalid_records():
    records = [
        FastaRecord("one", "one", "ACDEFGHIK"),
        FastaRecord("duplicate", "duplicate", "ACDEFGHIK"),
        FastaRecord("invalid", "invalid", "ACDEXGHIK"),
        FastaRecord("short", "short", "ACD"),
    ]
    cleaned, report = clean_positive_records(records, min_length=5, max_length=20)
    assert len(cleaned) == 1
    assert report["exact_duplicates_removed"] == 1
    assert report["invalid_or_noncanonical"] == 1
    assert report["outside_length_range"] == 1


def test_negative_fragments_are_deterministic_and_length_matched():
    parents = [
        FastaRecord("p1", "p1", "ACDEFGHIKLMNPQRSTVWY" * 3),
        FastaRecord("p2", "p2", "WYVTSRQPNMLKIHGFEDCA" * 3),
    ]
    target_lengths = [8, 10, 12]
    first = sample_negative_fragments(
        parents, target_lengths, forbidden_sequences=set(), random_state=7
    )
    second = sample_negative_fragments(
        parents, target_lengths, forbidden_sequences=set(), random_state=7
    )
    assert first == second
    assert sorted(len(row["sequence"]) for row in first) == sorted(target_lengths)
    assert len({row["sequence"] for row in first}) == len(first)


def test_forbidden_positive_fragment_is_not_used():
    parent = FastaRecord("p1", "p1", "ACDEFGHIKLMNPQRSTVWY")
    fragments = sample_negative_fragments(
        [parent], [5], forbidden_sequences={"ACDEF"}, random_state=1
    )
    assert fragments[0]["sequence"] != "ACDEF"
