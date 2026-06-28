# Encoder baselines

`train_hf_audit_classifier.py` trains biomedical or clinical encoder classifiers for the same structured label-audit task. Each encoder receives the same task context as the generative auditor: report text, target finding, finding definition, and candidate binary label.

The manuscript compared six encoder baselines: PubMedBERT/BiomedBERT, BioClinicalBERT, RadBERT, BioMed-RoBERTa, BioELECTRA, and Clinical-Longformer.
