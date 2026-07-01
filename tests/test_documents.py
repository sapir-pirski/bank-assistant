from app.documents import load_markdown_chunks


def test_load_markdown_chunks_preserves_heading_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "policy.md").write_text(
        "# Bank Guide\n\n"
        "Intro paragraph.\n\n"
        "## Cards\n\n"
        "Card policy details for testing.\n\n"
        "### Abroad\n\n"
        "Use the card abroad according to the policy.",
        encoding="utf-8",
    )

    chunks = load_markdown_chunks(data_dir, chunk_size=120, overlap=0)

    assert chunks
    assert any(chunk.heading == "Bank Guide > Cards > Abroad" for chunk in chunks)
    assert all(chunk.source == "data/policy.md" for chunk in chunks)
    assert all(chunk.metadata["source"] == chunk.source for chunk in chunks)
    assert all("heading" in chunk.metadata for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_load_markdown_chunks_splits_long_text_with_overlap(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    long_text = " ".join(f"word{i}" for i in range(80))
    (data_dir / "policy.md").write_text(f"# Guide\n\n{long_text}", encoding="utf-8")

    chunks = load_markdown_chunks(data_dir, chunk_size=90, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].id != chunks[1].id
    assert chunks[0].text[-20:].strip().split()[-1] in chunks[1].text
