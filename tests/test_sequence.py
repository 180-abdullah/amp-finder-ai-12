import pytest

from amp_finder.sequence import SequenceValidationError, normalize_sequence, sequence_sha256


def test_normalize_plain_and_fasta_input():
    assert normalize_sequence(" acd-efg \n") == "ACDEFG"
    assert normalize_sequence(">example\nACD EFG\n") == "ACDEFG"


def test_noncanonical_residue_is_rejected():
    with pytest.raises(SequenceValidationError, match="Invalid symbol"):
        normalize_sequence("ACDEXG")


def test_length_limits_are_enforced():
    with pytest.raises(SequenceValidationError, match="too short"):
        normalize_sequence("ACD")
    with pytest.raises(SequenceValidationError, match="too long"):
        normalize_sequence("A" * 201)


def test_sequence_hash_is_stable_and_not_the_sequence():
    digest = sequence_sha256("ACDEFG")
    assert digest == sequence_sha256("ACDEFG")
    assert len(digest) == 64
    assert "ACDEFG" not in digest
