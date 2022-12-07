This repository is a Multivariate Time Series Anomaly Detection toolkit contains state-of-the-art methods with a unified and easy-to-use interface.

Multivariate time series are a group of inherently correlated time series. For example, in the area of manufacturing industry and Information Technology (IT) systems, an entity (e.g., a physical machine or software service) is generally equipped with a monitoring mechanism to ensure its security or reliability.

In contrast to anomaly detection on single time series, extensive recent studies indicate that dependency hidden in MTS is of great importance for accurate anomaly detection, namely, the anomaly detector should consider the MTS as a whole. To this end, state-of-the-art methods have resort to deep learning-based methods to capture the dependency for more accurate anomaly detection.


## Models integrated in this Repo.


|       Model       | Paper reference                                              |
| :---------- | :----------------------------------------------------------- 

| MSCRED      | [AAAI'19] A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data. |
| :---------- | :----------------------------------------------------------- |
| MTAD-GAT |     **[ICDM'2020]** Multivariate Time-series Anomaly Detection via Graph Attention Networks |
| GDN |      **[AAAI 2021]** Graph Neural Network-Based Anomaly Detection in Multivariate Time Series |
| GLUE |       **[ICLR'2021]** Learning Graph Neural Networks for Multivariate Time Series Anomaly Detection |




### Datasets 

The following datasets are kindly released by different institutions or schools. Raw datasets could be downloaded or applied from the link right behind the dataset names. The processed datasets can be found [here](https://drive.google.com/drive/folders/1NEGyB4y8CvUB8TX2Wh83Eas_QHtufGPR?usp=sharing)⬇️ (SMD, SMAP, and MSL).

- Server Machine Datase (**SMD**) [Download raw datasets⬇️](https://github.com/NetManAIOps/OmniAnomaly.git)

  > Collected from a large Internet company containing a 5-week-long monitoring KPIs of 28 machines. The meaning for each KPI could be found [here](https://github.com/NetManAIOps/OmniAnomaly/issues/22).

- Soil Moisture Active Passive satellite (**SMAP**) and Mars Science Laboratory rovel (**MSL**) [Download raw datasets⬇️](link)

  > They are collected from the running spacecraft and contain a set of telemetry anomalies corresponding to actual spacecraft issues involving various subsystems and channel types.

- Secure Water Treatment (**WADI**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  >WADI is collected from a real-world industrial water treatment plant, which contains 11-day-long multivariate KPIs. Particularly, the system is in a normal state in the first seven days and is under attack in the following four days.

- Water Distribution (**SWAT**) [Apply here\*](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  > An extended dataset of SWAT. 14-day-long operation KPIs are collected when the system is running normally and 2-day-long KPIs are obtained when the system is in attack scenarios.

\* WADI and SWAT datasets were released by iTrust, which should be individually applied. One can request the raw datasets and preprocess them with our preprocessing scripts.

