"""Drive the Streamlit app headlessly with Streamlit's own test runner."""

from __future__ import annotations

import shutil

import pytest

import indexer

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

from conftest import SOURCE_DIRECTORY  # noqa: E402

APP_PATH = str(SOURCE_DIRECTORY / "app.py")


def open_app(index_directory):
    app = streamlit_testing.AppTest.from_file(APP_PATH, default_timeout=60).run()
    sidebar_input = next(t for t in app.text_input if t.label == "Index directory")
    sidebar_input.input(str(index_directory)).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_missing_index_shows_instructions(tmp_path):
    app = streamlit_testing.AppTest.from_file(APP_PATH, default_timeout=60).run()
    next(t for t in app.text_input if t.label == "Index directory").input(
        str(tmp_path / "nowhere")
    ).run()

    assert any("No index found" in e.value for e in app.error)
    assert not app.exception


def test_text_search_with_both_backends(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    app = open_app(index_directory)

    query = next(t for t in app.text_input if t.label.startswith("Describe"))
    query.input("a dog").run()

    for backend, label in (("flat", "exact"), ("hnsw", "HNSW")):
        app.radio[0].set_value(backend).run()
        button(app, "Search").click().run()
        assert not app.exception, [e.value for e in app.exception]
        assert any(f"({label})" in c.value for c in app.caption)


def test_index_type_switch_disabled_without_graph(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    app = open_app(index_directory)

    assert app.radio[0].disabled
    assert any("No HNSW index" in c.value for c in app.caption)


def test_duplicates_tab(image_directory, index_directory):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "copy.png")
    indexer.build_index(image_directory, index_directory)
    app = open_app(index_directory)

    button(app, "Find duplicates").click().run()

    assert not app.exception, [e.value for e in app.exception]
    assert any("Group 1 · 2 images" in h.value for h in app.subheader)


def test_duplicates_tab_with_none_found(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    app = open_app(index_directory)

    button(app, "Find duplicates").click().run()

    assert any("No near-duplicates" in s.value for s in app.success)
