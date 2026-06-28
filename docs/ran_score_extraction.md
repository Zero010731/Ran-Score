# Ran Score extraction

`scripts/run_qwen_extractor.py` applies a Qwen3-14B extractor to radiology report text and returns a standardized 21-label finding vector. The labels are used to compute finding-level macro-F1 between generated reports and reference reports.

The public script is parameterized and does not include private reports, MIMIC-CXR text, or local filesystem paths.
