# RC-8: Set up environment configs

**Status:** closed
**Epic:** EPIC-1
**Priority:** medium

## Description

Create environment-specific configuration files for dev, staging, and production.

## Tasks

- [x] Create config/dev.yaml
- [x] Create config/staging.yaml
- [x] Create config/prod.yaml
- [x] Define model, tool, memory, quality settings per environment

## Acceptance Criteria

- [x] All three config files exist
- [x] Settings vary appropriately by environment
- [x] YAML is valid and well-structured

## Notes

Created in config/:
- dev.yaml - DEBUG logging, 3 parallel researchers
- staging.yaml - INFO logging, 4 parallel researchers
- prod.yaml - WARNING logging, 5 parallel researchers, long-term memory enabled
