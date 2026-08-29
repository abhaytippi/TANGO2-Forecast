# Models

Fitted model artifacts are written here.

The application trains its classifier at startup and caches it in memory rather
than shipping a serialised model, for one reason: a serialised model fitted to
synthetic data could be loaded later without its provenance and mistaken for a
model fitted to real patients. Training in process keeps the provenance flag
attached to the object that produces every prediction.

If you serialise a model for deployment, serialise the whole `DiagnosticModel`
including its `provenance` field, never the bare scikit-learn estimator.
