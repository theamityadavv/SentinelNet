# Reflection on Network Traffic Analysis

### Observations and Challenges in Network Intrusion Detection

Analyzing the NSL-KDD dataset reveals clear patterns in network traffic. Most traffic is **normal**, while a significant portion corresponds to **Denial-of-Service (DoS)** attacks, which occur frequently and in large volumes. Probe attacks like scanning and reconnaissance appear moderately, whereas **Remote-to-Local (R2L)** and **User-to-Root (U2R)** attacks are rare. This imbalance shows that certain attack types dominate network traffic, while others are subtle and sparse. Features such as `src_bytes`, `dst_bytes`, and protocol types help distinguish traffic categories, and grouping attacks into broader categories (dos, probe, r2l, u2r) provides a clearer overview of threat distribution.

Despite identifiable patterns, detecting intrusions is challenging due to several factors. Firstly, attacks constantly evolve, making historical data sometimes insufficient for predicting new threats. Secondly, subtle attacks like R2L or U2R often resemble normal traffic, creating **high false-negative rates**. Thirdly, real network environments produce huge volumes of traffic, requiring efficient real-time analysis. Finally, some features are categorical or sparse, demanding advanced preprocessing and feature engineering.

Overall, network intrusion detection requires **adaptive and intelligent models** that can generalize across evolving attack patterns while maintaining low false positives and high accuracy.
