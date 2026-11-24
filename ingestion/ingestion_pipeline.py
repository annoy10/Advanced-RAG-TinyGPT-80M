from workflows.ingestion_graph import prepare_sources

class IngestionPipeline:
    def run(self, sources):
        """Load, normalize, deduplicate, chunk, validate, and enrich source records."""
        return prepare_sources(sources)
