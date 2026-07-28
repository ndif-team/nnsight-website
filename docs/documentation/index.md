# API Reference

Auto-generated reference for the `nnsight` package, organized to mirror the
source tree. Most users only need the model classes and the tracing API; the
rest documents the internals.

## Sections

- **[intervention](intervention/index.md)** — the tracing/interleaving engine: `Envoy`, the tracer and invoker, batching, caching, gradients, editing, and the local/remote backends.
- **[modeling](modeling/index.md)** — model wrappers: `NNsight`, `TransformersModel`, `LanguageModel`, `VisionLanguageModel`, `DiffusionModel`, the loadable/meta/remotable mixins, and the vLLM backend.
- **[tracing](tracing/index.md)** — the model-agnostic tracing layer the intervention engine builds on (base `Tracer`, `Backend`, globals, hints).
- **[schema](schema/index.md)** — the config singleton and the request/response wire models.
- **[ndif](ndif.md)** — top-level NDIF helpers: `login`, `status`, `compare`, `register`.
- **[util](util.md)** — general utilities.
