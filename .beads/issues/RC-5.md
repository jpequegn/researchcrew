# RC-5: Implement read_url tool

**Status:** closed
**Epic:** EPIC-1
**Priority:** high

## Description

Implement the read_url tool to fetch and extract content from web pages.

## Tasks

- [x] Use httpx for HTTP requests
- [x] Use BeautifulSoup for HTML parsing
- [x] Remove script/style/nav elements
- [x] Truncate long content
- [x] Handle errors gracefully

## Acceptance Criteria

- [x] Tool fetches URLs successfully
- [x] HTML is parsed and cleaned
- [x] Content is truncated to reasonable length
- [x] Errors return helpful messages

## Notes

Implemented in tools/search.py with:
- httpx for requests
- BeautifulSoup for parsing
- 10,000 char content limit
- Error handling for HTTP and parsing errors
