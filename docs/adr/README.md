# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) that document important architectural decisions made for the IoT Edge Graph Anomaly Detection system.

## What are ADRs?

Architecture Decision Records are short text documents that capture a single architectural decision. They help teams:
- Understand the context behind important decisions
- Track the evolution of the architecture over time
- Onboard new team members efficiently
- Avoid repeating past discussions

## ADR Format

We use a lightweight ADR format with the following sections:

```markdown
# ADR-XXXX: [Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](0001-lstm-autoencoder-architecture.md) | LSTM Autoencoder for Temporal Anomaly Detection | Accepted | 2025-08-02 |
| [ADR-0002](0002-container-deployment-strategy.md) | Container-Based Edge Deployment | Accepted | 2025-08-02 |
| [ADR-0003](0003-opentelemetry-observability.md) | OpenTelemetry for Observability | Accepted | 2025-08-02 |
| [ADR-0004](0004-pytorch-ml-framework.md) | PyTorch as Primary ML Framework | Accepted | 2025-08-02 |

## Creating New ADRs

1. Copy the [template](template.md) to a new file
2. Use the next sequential number: `XXXX-descriptive-title.md`
3. Fill in all sections thoroughly
4. Update this README with the new ADR entry
5. Create a pull request for review

## ADR Lifecycle

- **Proposed**: Initial draft, under discussion
- **Accepted**: Decision approved and implemented
- **Deprecated**: No longer recommended, but may still be in use
- **Superseded**: Replaced by a newer ADR (link to replacement)

## Guidelines

- Keep ADRs focused on a single decision
- Write for future team members who weren't part of the original discussion
- Include relevant context, alternatives considered, and trade-offs
- Update ADRs when decisions change (don't delete historical records)
- Reference related ADRs when applicable

---

**ADR Process Owner**: Architecture Team  
**Last Updated**: 2025-08-02