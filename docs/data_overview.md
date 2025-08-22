# NSL-KDD Dataset Overview

## Dataset Source

The NSL-KDD dataset is a benchmark dataset for network intrusion detection.
**Source:** [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)

**Files Used:**

* `KDDTrain+.txt` – Training dataset

---

## Dataset Description

The dataset contains **125,973 network connection records** with **43 columns**. Each record describes a connection with numeric and categorical features and is labeled as **normal** or a specific **attack type**.

---

## Columns / Features

1. `duration`
2. `protocol_type`
3. `service`
4. `flag`
5. `src_bytes`
6. `dst_bytes`
7. `land`
8. `wrong_fragment`
9. `urgent`
10. `hot`
11. `num_failed_logins`
12. `logged_in`
13. `num_compromised`
14. `root_shell`
15. `su_attempted`
16. `num_root`
17. `num_file_creations`
18. `num_shells`
19. `num_access_files`
20. `num_outbound_cmds`
21. `is_host_login`
22. `is_guest_login`
23. `count`
24. `srv_count`
25. `serror_rate`
26. `srv_serror_rate`
27. `rerror_rate`
28. `srv_rerror_rate`
29. `same_srv_rate`
30. `diff_srv_rate`
31. `srv_diff_host_rate`
32. `dst_host_count`
33. `dst_host_srv_count`
34. `dst_host_same_srv_rate`
35. `dst_host_diff_srv_rate`
36. `dst_host_same_src_port_rate`
37. `dst_host_srv_diff_host_rate`
38. `dst_host_serror_rate`
39. `dst_host_srv_serror_rate`
40. `dst_host_rerror_rate`
41. `dst_host_srv_rerror_rate`
42. `label` – Specific attack type or normal
43. `difficulty` – Detection difficulty score

---

## Attack Categories

| Category | Sample Attacks                                                                |
| -------- | ----------------------------------------------------------------------------- |
| DOS      | neptune, smurf, pod, back, teardrop                                           |
| Probe    | satan, ipsweep, nmap, portsweep                                               |
| R2L      | ftp\_write, guess\_passwd, imap, phf, spy, multihop, warezclient, warezmaster |
| U2R      | buffer\_overflow, loadmodule, perl, rootkit                                   |
| Normal   | normal                                                                        |

**Distribution:**

* `normal`: 67,343
* `dos`: 45,927
* `probe`: 11,656
* `r2l`: 995
* `u2r`: 52

---

## Dataset Stats

* **Shape:** `(125973, 44)` (features + label + attack category)
* **Numeric stats:** Mean, Std, Min, Max, Quartiles available for numeric features
* **No missing values** in the dataset

---

## Visualizations

* Bar chart of attack categories: shows count per category
* Grouped counts by `attack_category` and `label`

---

## Notes

* Dataset has **mixed numeric and categorical features**
* Used for training ML models for **network intrusion detection**
