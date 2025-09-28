# Data Overview

## Dataset Sources
- **NSL-KDD**: [Download Link](https://www.unb.ca/cic/datasets/nsl.html)
- **CICIDS2017**: [Download Link](https://www.unb.ca/cic/datasets/ids-2017.html)

## NSL-KDD Schema
- Features: 41 (e.g., duration, protocol_type, service, flag, src_bytes, dst_bytes, …)
- Label: attack type (normal or different attack categories)

## Summary
- Train Dataset Rows: ~125,973
- Test Dataset Rows: ~22,544
- Unique Attack Labels: 23
- Top 5 Frequent Attacks: neptune, normal, smurf, satan, ipsweep

## Reflection

When analyzing the NSL-KDD dataset, one key observation is that network traffic is highly imbalanced. Certain attacks such as *neptune* and *smurf* appear very frequently, while others such as *perl* or *spy* occur only a handful of times. This imbalance reflects real-world traffic, where most events are normal or dominated by a few common types of attacks. Another pattern is that some features, like protocol type and service, strongly influence the distribution of attacks, highlighting how attackers often target specific services.

Detecting intrusions is challenging because malicious traffic can closely resemble normal traffic. Attackers continuously adapt their methods, introducing novel threats that may not match historical patterns. Moreover, high-dimensional network data with 40+ features makes it difficult to identify subtle anomalies. Many attacks are rare, so machine learning models may fail to generalize. Additionally, modern networks generate huge volumes of traffic, making real-time detection computationally expensive.

Overall, intrusion detection requires handling imbalanced datasets, evolving attack patterns, and distinguishing between very similar normal and abnormal traffic. These challenges motivate the need for advanced anomaly detection, AI-driven approaches, and continuous dataset updates.
