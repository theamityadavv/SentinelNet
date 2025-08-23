# Network Intrusion Detection Datasets Documentation

## 1. NSL-KDD Dataset

**Origin**\
- File Used: `KDDTrain+.txt`\
- Overview: The NSL-KDD dataset is an enhanced and cleaner version of
the original KDD Cup 1999 dataset, frequently applied in academic
research on intrusion detection.

**Dataset Properties**\
- Total Instances: \~125,973 records\
- Features: 42 attributes\
- Target Column: Located in the final column

**Classes**\
- Distinct categories: `normal`, `neptune`, `smurf`, `satan`, `ipsweep`,
plus several others\
- Most common classes:\
1. neptune\
2. smurf\
3. normal\
4. satan\
5. ipsweep

------------------------------------------------------------------------

## 2. CICIDS2017 Dataset

**Origin**\
- File Example: `DoS-Wednesday-no-metadata.parquet`\
- Overview: CICIDS2017 simulates real-world traffic, containing both
benign and malicious flows, making it valuable for contemporary
intrusion detection benchmarking.

**Dataset Properties**\
- Size: Over 2.8 million flow entries\
- Features: More than 80 attributes\
- Target Column: `Label`

**Classes**\
- Unique categories include: DoS Hulk, DoS GoldenEye, DoS Slowloris,
DDoS, PortScan, FTP-Patator, SSH-Patator, and Normal traffic\
- Most common attack classes:\
1. DoS Hulk\
2. DoS GoldenEye\
3. DoS Slowloris\
4. DDoS\
5. PortScan

------------------------------------------------------------------------

## Summary

-   **NSL-KDD**: Compact, widely cited, useful for quick testing and
    benchmarking IDS methods.\
-   **CICIDS2017**: Larger, more modern, offering richer and more
    realistic network scenarios.

These datasets are essential resources for building, evaluating, and
comparing Network Intrusion Detection Systems (NIDS).
