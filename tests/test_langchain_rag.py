import pytest

def test_langchain_rag_module_imports_without_eager_langchain_dependency():
    from generation.langchain_rag import LangChainRAGWorkflow
    assert LangChainRAGWorkflow is not None

def test_langchain_rag_requires_langchain_packages():
    try:
        import langchain_core  # noqa
    except ImportError:
        from generation.langchain_rag import LangChainRAGWorkflow
        with pytest.raises(RuntimeError, match="LangChain packages are missing"):
            LangChainRAGWorkflow(object())
