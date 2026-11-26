# Inter-patient MIT-BIH heartbeat classification: the de Chazal protocol, RR / rhythm features, and the baselines a new method is compared with

Notes compiled 2026-11-26. Reading status is marked per source: **[full text read]**, **[abstract only]**, **[secondhand: numbers copied from another paper's comparison table]**. Venue flags: **[arXiv-only]**, **[MDPI]**, **[Hindawi]**, **[conference]**. "(c)" means I computed the value (usually F1 = 2·Se·+P/(Se+P)) from numbers the paper reports. It is not a number the paper prints.

Main primary source: de Chazal, O'Dwyer & Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features", IEEE TBME 51(7):1196-1206, July 2004. DOI 10.1109/TBME.2004.827359. Read in full from the author-repository PDF (TCD TARA, green OA, archived copy). Citation count: 1,912 on OpenAlex and 1,740 on Semantic Scholar, as of 2026-11-26.
- Open copy: [TARA handle](http://hdl.handle.net/2262/17623); [archived PDF](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

---

## Q1. The de Chazal et al. 2004 protocol: record lists, exclusions, AAMI mapping, beat counts, features, and the paper's own ablation

### Takeaway
The protocol excludes the four paced records (102, 104, 107, 217) and splits the remaining 44 records 22/22 into DS1 (train and model selection) and DS2 (test only). Classes follow the five AAMI EC57:1998 classes. The paper has 4 RR features: pre-RR, post-RR, a whole-record average RR, and a local average over the 10 surrounding RR intervals. **Every one of its 8 feature sets and 12 configurations contains all four RR features.** So the paper never measures how much RR adds, for S or for any other class. Its ablation varies only lead, morphology sampling, amplitude scaling and use of wave segmentation. Its key methodological result is a different one: in record-wise cross-validation the same classifier scores 7.1 accuracy points lower than in beat-wise cross-validation (79.0 vs 86.1 %).

### Cited Findings
**Data and split** (full text read; [de Chazal 2004, archived PDF](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf))
- **Recordings and leads:** 48 two-lead 30-min recordings at 360 Hz, band-pass filtered 0.1–100 Hz. Lead A is modified lead II in 45 recordings (V5 in the other 3). Lead B is V1 in 40 recordings (II, V2, V4 or V5 in the rest).
- **Paced records removed:** "In agreement with the AAMI recommended practice, the four recordings containing paced beats were removed". Table II footnote: "The four paced recordings (102, 104, 107 and 217) were not included in dataset 1 or 2."
- **DS1 (Table II footnote):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230.
- **DS2:** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.
- **How the split was used:** each dataset has "approximately 50 000 heartbeats" from 22 recordings "with the same approximate proportion of beat types". DS1 was used to choose among candidate classifiers. DS2 was used only for the final evaluation, and the paper says it "was not used at any point in the development of the classifier."
- **Beat positions:** taken from the database's annotated fiducial points ("first obtained automatically and then manually corrected"). The method uses no beat detector. The authors note that automatic detection would add errors to the RR features.

**AAMI class mapping (Table I, verbatim content)**
- **N:** normal (NOR), left bundle branch block (LBBB), right bundle branch block (RBBB), atrial escape (AE), nodal (junctional) escape (NE).
- **S (SVEB):** atrial premature (AP), aberrated atrial premature (aAP), nodal (junctional) premature (NP), supraventricular premature (SP).
- **V (VEB):** premature ventricular contraction (PVC), ventricular escape (VE).
- **F:** fusion of ventricular and normal (fVN).
- **Q:** paced (P), fusion of paced and normal (fPN), unclassified (U).
- Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Beat counts (Table II, copied from the paper; class sums are my arithmetic)**

| Heartbeat type | NOR | LBBB | RBBB | AP | aAP | NP | SP | PVC | fVN | AE | NE | VE | P | fPN | U | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAMI class | N | N | N | S | S | S | S | V | F | N | N | V | Q | Q | Q | |
| Full DB | 75054 | 8074 | 7259 | 2544 | 150 | 83 | 2 | 7129 | 803 | 16 | 229 | 106 | 7028 | 982 | 33 | 109492 |
| DS1 | 38104 | 3949 | 3783 | 808 | 100 | 32 | 2 | 3682 | 415 | 16 | 16 | 105 | 0 | 0 | 8 | 51020 |
| DS2 | 36444 | 4125 | 3476 | 1736 | 50 | 51 | 0 | 3220 | 388 | 0 | 213 | 1 | 0 | 0 | 7 | 49711 |

- **Class totals (c):**
  - DS1: N 45,868 · S 942 · V 3,787 · F 415 · Q 8 = 51,020.
  - DS2: N 44,258 · S 1,837 · V 3,221 · F 388 · Q 7 = 49,711.
  - The text agrees: "there were 942 SVEBs in the training data … 3787 VEBs". Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)
- **A commonly cited, slightly different count:** DS1 N 45,866 / S 944 / V 3,788 / F 415 / Q 8 = 51,021 and DS2 44,259 / 1,837 / 3,221 / 388 / 7 = 49,712. It is printed as "as outlined by de Chazal et al." in [Silva et al. 2025 systematic review, Table 3](https://arxiv.org/abs/2503.07276) **[arXiv-only]**, and the DS2 N = 44,259 appears in [Cao et al. Table 2](https://arxiv.org/abs/2206.14200) **[arXiv-only]**. It does **not** match de Chazal's own Table II (DS1 differs by +2 N, +2 S, +1 V; DS2 by +1 N). Other papers' beat extraction gives yet other counts. Examples:
  - Mousavi & Afghah: DS1 N 45,796 / S 941 / V 3,780 ([arXiv 1812.07421](https://arxiv.org/abs/1812.07421)).
  - Mommen et al. 2026: S 3,778 train + 3,208 test, because e and j beats are mapped to S ([arXiv 2609.32854](https://arxiv.org/abs/2609.32854)).
- **Where the DS2 S and F beats are (Table VII(a), per-record counts):**
  - S: record 232 holds **1,382 of the 1,837 DS2 S beats (75.2 %, c)** and record 222 holds 209.
  - F: record 213 holds 362 of the 388 DS2 F beats.
  - V: records 200 (826), 233 (831), 221 (396), 213 (220), 210 (195) and 214 (256) hold most of the DS2 V beats.
  - Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Features (Section IV-D and Table III, verbatim definitions)**
- **RR intervals** (shared by both leads; "fiducial point intervals"):
  - "The pre-RR-interval was the RR-interval between a given heartbeat and the previous heartbeat."
  - "The post-RR-interval was the RR-interval between a given heartbeat and the following heartbeat."
  - "The average RR-interval was the mean of the RR-intervals for a recording and had the same value for all heartbeats in a recording."
  - "the local average RR-interval was determined by averaging the RR-intervals of the ten RR-intervals surrounding a heartbeat."
  - Note: post-RR and both averages use samples after the current beat.
- **Heartbeat intervals** (per lead): QRS duration (QRS onset to offset); T-wave duration (QRS offset to T offset); a Boolean P-wave flag. All come from the ecgpuwave segmenter of Laguna et al.
- **Morphology, 8 groups:**
  - Morphology 1A/1B: 10 samples between QRS onset and offset plus 9 samples between QRS offset and T offset, both uniformly sampled.
  - 2A/2B: the same after scaling the whole signal to unit standard deviation.
  - 3A/3B: fixed windows. 10 samples at 60 Hz from FP−50 ms to FP+100 ms, plus 8 samples at 20 Hz from FP+150 ms to FP+500 ms.
  - 4A/4B: 3A/3B after scaling.
- **Preprocessing:** 200-ms then 600-ms median filters for baseline removal, then a 12-tap FIR low-pass filter (3-dB point 35 Hz).
- Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Feature sets and configurations**
- FS1 has "26 features including the RR features and the features from lead A requiring the determination of the heartbeat segmentation points". FS2 swaps in Morphology 2A. FS3 is 22 features "including the RR features" plus Morphology 3A. FS4 uses 4A. FS5–FS8 are the lead-B versions.
- Configurations I–VIII each use one feature set with one linear discriminant (LD). IX–XII combine FS1+FS5, FS2+FS6, FS3+FS7 and FS4+FS8 with a product rule on the class posteriors.
- **No configuration omits the RR features, so the paper gives no "with vs without RR" number.** Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Classifier and training details**
- Linear discriminants with a weighted likelihood: each class contributes "the equivalent of 400 examples".
- Priors are 10/41 for N, S, V and F, and 1/41 for Q.

**Selection results on DS1 (Table VI, multiway accuracy / specificity / Se for SVEB, VEB, F, Q)**
- (a) Configuration I, **beat-wise** 22-fold CV (DD2): Acc 86.1, Spec 87.0, Se S 77.6, V 76.4, F 87.0, Q 0.0.
- (b) Configuration I, **record-wise** 22-fold CV (DD1): 79.0 / 80.6 / 59.9 / 65.3 / 74.5 / 25.0.
- Other single-lead configurations (DD1):
  - II: 76.9 / 78.3 / 44.5 / 67.3 / 85.3 / 25.0
  - III: 81.8 / 86.5 / 56.5 / 39.1 / 16.1 / 12.5
  - IV: 77.8 / 80.9 / 32.6 / 57.0 / 23.1 / 12.5
  - V: 74.1 / 76.6 / 32.6 / 55.6 / 62.2 / 12.5
  - VI: 74.9 / 77.2 / 29.8 / 60.7 / 51.6 / 0.0
  - VII: 83.0 / 88.1 / 30.4 / 39.9 / 32.1 / 0.0
  - VIII: 78.6 / 82.5 / 29.0 / 42.8 / 80.7 / 0.0
- Multi-lead configurations (DD1):
  - **IX (selected): 84.5 / 86.7 / 53.3 / 67.3 / 71.6 / 12.5**
  - X: 82.9 / 84.8 / 46.1 / 69.1 / 83.6 / 0.0
  - XI: 86.2 / 91.2 / 54.7 / 40.0 / 26.0 / 0.0
  - XII: 84.2 / 87.3 / 45.3 / 58.7 / 64.8 / 0.0
- Interpretation in the paper: lead-B-only sets give much lower S sensitivity (29–33 %) than lead-A sets (32–60 %). Unscaled morphology plus segmentation points (FS1) was best for balanced sensitivities.
- Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Final DS2 result (configuration IX, AAMI "gross" statistics)**
- SVEB: Se **75.9**, +P **38.5**, FPR 4.7.
- VEB: Se **77.7**, +P **81.9**, FPR 1.2.
- Table VIII confusion matrix, rows = reference N, S, V, F, Q; columns = predicted n, s, v, f, q:
  - N: 38444, 1904, 303, 3509, 98
  - S: 173, 1395, 252, 16, 1
  - V: 117, 321, 2504, 176, 103
  - F: 33, 1, 7, 347, 0
  - Q: 4, 0, 3, 0, 0
- Derived from that matrix (c): multiway accuracy **85.9 %**; N Se 86.9 / +P 99.2; F Se 89.4 / +P 8.6; SVEB F1 51.1; VEB F1 79.7.
- Main errors named in the paper: "1904 normal (N) beats were misclassified as SVEB (S) beats, and … 3509 N beats were misclassified as fusion (F) beats."
- Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)
- **Conflict:** [Garcia et al. 2017, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) lists de Chazal's accuracy as 83. The README of [Mondéjar-Guerra's code](https://github.com/mondejar/ecg-classification) lists 0.862. The Table VIII confusion matrix gives 85.9.

**AAMI metric conventions used by the paper (Table V)**
- The VEB statistics "do not reward or penalize a classifier for the classification of ventricular fusion (F) or unknown beats (Q) as VEBs".
- The SVEB statistics ignore Q predicted as SVEB.
- Aggregates are "gross" statistics: every beat is weighted equally.
- Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

**Database facts (PhysioNet)**
- The database is "48 approximately half-hour recordings from 47 subjects." The PhysioNet page now states "Records 201 and 202 originate from the same subject" and "For subject-independent evaluation, records 201 and 202 should be assigned to the same partition". Its source is the MIT-BIH Arrhythmia Database Directory. [PhysioNet mitdb 1.0.0](https://physionet.org/content/mitdb/1.0.0/)
- In the de Chazal split, 201 is in DS1 and 202 is in DS2. Also noted by [Silva et al. 2025](https://arxiv.org/abs/2503.07276) **[arXiv-only]**: "records 201 and 202, originating from the same patient, are placed in different datasets".

### Inferences
- de Chazal's average RR uses the whole record and its local average is centred, so both look up to ~5 beats ahead. Post-RR looks one beat ahead. The "reference" feature set is therefore not causal.
- The paper cannot answer "how much do RR features add for S beats". That number has to come from later ablations (Q3/Q5 below: Cheng et al. 2020, Farag 2023).
- Because 75 % of DS2 S beats come from record 232, a DS2 SVEB Se/+P is mostly a statement about one patient. Small methodological changes can swing it a lot.

### Gaps
- The paper gives no per-feature-group RR ablation. I found no later reproduction of de Chazal's exact LD pipeline without RR features.
- I did not find the source of the widely reproduced "45,866 / 944 / 3,788 …" counts. They are probably from a re-extraction with a different handling of first and last beats, but this is unverified.

---

## Q2. Why the random beat split (intra-patient, e.g. Kachuee et al. 2018's 187-sample Kaggle set) is inflated, and same-model evidence of the gap

### Takeaway
Under a random beat split, beats of the same patient appear in both training and test sets. Those beats are highly dependent, so the model is rewarded for memorising patients. de Chazal showed this inside the original paper: the same classifier scored 86.1 % with beat-wise folds and 79.0 % with record-wise folds, with S Se 77.6 vs 59.9. Later same-model comparisons show drops of 13–25 accuracy points. The S class collapses when Kachuee's CNN is re-trained on DS1 and tested on DS2: S recall goes from 82.6 % to 0 %.

### Cited Findings
- **de Chazal 2004, same classifier (configuration I) under two divisions of DS1:**
  - Beat-wise folds (DD2): Acc 86.1, Spec 87.0, Se S 77.6, V 76.4, F 87.0.
  - Record-wise folds (DD1): 79.0, 80.6, 59.9, 65.3, 74.5.
  - The paper: "The multiway accuracy estimated using DD2 was 86.1% which was 7.1% higher than the same result estimated using DD1 … dividing data on a beat basis results in optimistically biased classifications results. This scheme is biased as intrasubject beats can be highly dependent and hence dependent beats may appear in the training and testing sets. This data division scheme has been previously favored by other investigators … and should be avoided."
  - Source: [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)
- **Ye, Kumar & Coimbra 2012 (IEEE TBME 59(10)), same method under both evaluations:** "overall accuracy … of 99.3% (99.7% with 2.4% rejection) in the 'class-oriented' evaluation and an accuracy of 86.4% in the 'subject-oriented' evaluation". The method is wavelet plus ICA morphology with RR "dynamic features", an SVM, and fusion of both leads. [PubMed 22907960](https://pubmed.ncbi.nlm.nih.gov/22907960/) **[abstract only]**. Its DS2 per-class numbers are in the Q3 table.
- **Cao et al. (arXiv 2206.14200, UCLA) re-implemented Kachuee et al. 2018 and Romdhane et al. under both splits** **[arXiv-only; full text read]**:
  - Kachuee CNN, intra-patient (their 80/20 random split): **Acc 93.1**. Pre/Rec: N 98.4/94.3, S 38.1/82.6, V 96.6/82.3, F 26.6/93.8, Q 98.3/92.6. The literature reports 93.4.
  - Same model, DS1 to DS2: **Acc 81.2**. Pre/Rec: N 94.4/84.5, **S 0.0/0.0**, V 30.9/92.4, F 1.0/1.3.
  - Romdhane et al.: intra 82.7 vs inter 62.1.
  - The authors: "the testing set of intra-patient paradigm is susceptible to contamination … the intra-patient split paradigm is more likely to generate inflated and biased results".
  - On Kachuee specifically: "authors in [1] tried to mitigate the imbalance problem even in the 'test set' which is contrary to machine learning practice … the number of test samples in F class is equal to the entire class".
  - Source: [Cao et al., arXiv 2206.14200](https://arxiv.org/abs/2206.14200)
- **Classic methods evaluated both ways, as tabulated by Mousavi & Afghah 2019** **[conference (ICASSP 2019) + arXiv; secondhand for these rows]**. Intra-patient accuracy vs inter-patient (DS1 to DS2) accuracy:
  - Ye et al. (2010): 96.50 → 75.2, with S Se 72.4 → 3.2.
  - Yu & Chou (2008): 95.4 → 75.2, with S Se 73.8 → 1.8.
  - Song et al. (2005): 98.7 → 76.3, with S Se 86.4 → 27.0.
  - The inter-patient figures for these three are almost certainly the re-implementations of Luz & Menotti 2011. Mousavi cites only the original method papers, so this attribution is unverified.
  - Source: [Mousavi & Afghah, arXiv 1812.07421](https://arxiv.org/abs/1812.07421)
- **Luz & Menotti 2011 (EMBC; PMID 22255458):** "we analyze and criticize the results of some arrhythmia classification methods presented in the literature in terms of how the samples are chosen for training/testing the classifier and the impact this choice has on their performance … From our implementation, we also report new accuracies for these methods". [PubMed 22255458](https://pubmed.ncbi.nlm.nih.gov/22255458/) **[conference; abstract only — the numbers were not seen]**
- **Garcia et al. 2017 (Scientific Reports):**
  - Reporting only intra-patient results is "a serious problem", because models learn individual patients' characteristics. Their Fig. 1 shows heartbeats clustering by individual.
  - Intra-patient work reports >98 %. "Chen et al. reported 100% accuracy using only RR-interval information … Such results cannot be taken into consideration from a clinical point of view."
  - Source: [PMC5585360](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) (summary of the full text via fetch)
- **Silva et al. 2025 systematic review (UFOP, Luz is an author):** "De Chazal et al. demonstrated that including heartbeats from the same patient in both phases introduces bias … leading to inflated test performance (often approaching 100%)". Also: "many models are still evaluated using intra-patient methods, leading to inflated performance metrics". [arXiv 2503.07276](https://arxiv.org/abs/2503.07276) **[arXiv-only]**
- **Merdjanovska & Rashkovska 2023 (Scientific Reports 13:11682):** intra-patient CV is a "biased heartbeat selection" that "does not translate well to realistic scenarios". Their intra-vs-inter numbers are only in their GitHub repository and figures, not in the text. [PMC10356811](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356811)
- **Mousavi & Afghah's own model** reports 99.92 % intra-patient and 99.53 % inter-patient, which is almost no gap. That is anomalous next to every other same-model comparison above; see the Q3 and Q4 caveats. [arXiv 1812.07421](https://arxiv.org/abs/1812.07421)

### Inferences
- **The gap is largest for S beats.** S morphology is close to N, and the deciding cue is timing relative to that patient's own rhythm. With a random split, the classifier can match an S beat to near-identical S beats of the same patient, so S looks easy. Across patients, S Se falls to 0–60 % unless RR / rhythm context is given.
- Kachuee's 187-sample beats have no RR input and include only part of the next cycle. Its S collapse (82.6 → 0 %) is what a single-beat model with no rhythm information looks like under the inter-patient protocol.

### Gaps
- The Luz et al. 2016 CMPB survey (DOI 10.1016/j.cmpb.2015.12.008; 997 OpenAlex citations) was not accessible from this server. Its intra-vs-inter tables, if any, were not seen.
- The Luz & Menotti 2011 numbers were not seen directly.
- Kachuee et al. 2018 (ICHI 2018, conference) was not re-read here. The "beats start at the R-peak, 1.2×median RR, zero-padded to 187" description is from memory and was not re-verified in this session.

---

## Q3. Reported inter-patient (DS1 → DS2) results, 2004–2026: S and V class Se / +P / F1, and what input each method uses

### Takeaway
Fully automatic methods trained on DS1 and tested on DS2 cluster as follows:
- Classical RR + morphology methods (2004–2019): **S F1 ≈ 45–61** (Se 75–91 but +P only 31–53), V F1 ≈ 70–94.
- The best credible single-beat + RR deep models (2020–2023, journal venues): **S F1 ≈ 76–82, V F1 ≈ 90–95.** Examples are Niu 2020 (JBHI), Zahid 2023 (TBME), Wang 2021 (Entropy) and Farag 2023 (Sensors).
- Rule-based / abductive multi-beat interpretation (Teijeiro): S Se/+P ≈ 87/84 on DS2.

Almost every top result uses RR or rhythm context. Single-beat deep models without RR sit at **S F1 ≈ 0–31**. Every method that reports both settings improves S detection substantially when RR is added.

### Cited Findings
**Representative results table.** S = SVEB, V = VEB, on DS2. Input codes:
- **B** = single beat only
- **B+RR** = single beat plus RR features
- **MB** = multi-beat window or sequence

Marks for each RR / context field:
- **post** = uses the next beat's RR
- **rec-avg** = normalised by an average over the whole record
- **past-avg** = normalised by an average over past beats only

| # | Method (venue, year) | Flags | Input | S Se | S +P | S F1 | V Se | V +P | V F1 | Acc | Numbers from |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | de Chazal, O'Dwyer & Reilly (IEEE TBME 2004) | | B+RR (pre, post, rec-avg, centred local-10) + 2 leads | 75.9 | 38.5 | 51.1 (c) | 77.7 | 81.9 | 79.7 (c) | 85.9 (c) | [full text](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf) |
| 2 | Llamedo & Martínez (IEEE TBME 2011) | | B+RR (RR-series features + wavelet, 8 selected features, 2 leads) | 77 | 39 | 51.8 (c) | 81 | 87 | 83.9 (c) | 93 | [abstract](https://pubmed.ncbi.nlm.nih.gov/20729162/) |
| 3 | Mar et al. (IEEE TBME 2011) | | B+RR (SFFS-selected features, LD/MLP) | 83 | 33 | 47.2 (c) | 86 | 75 | 80.1 (c) | 89 | secondhand: [Garcia 2017 Tbl 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) |
| 4 | Ye, Kumar & Coimbra (IEEE TBME 2012) | | B+RR ("dynamic features"), 2 leads | 60 (61.02) | 53 (52.34) | 56.3 (c; Wang 56.34) | 81 (81.82) | 63 (61.45) | 70.9 (c; Wang 70.19) | 86.4 | [abstract](https://pubmed.ncbi.nlm.nih.gov/22907960/) for Acc; per-class secondhand from [Garcia 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) and, in parentheses, [Wang 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/) |
| 5 | Zhang et al. (Comput Biol Med 2014) | | B+RR (inter- and intra-beat intervals + morphology; one-vs-one SVM) | 79.06 | 35.98 | 49.46 | 85.48 | 92.75 | 88.96 | 86.66 | [abstract](https://pubmed.ncbi.nlm.nih.gov/24529208/); F1 from [Wang 2021 Tbl](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/) |
| 6 | Lin & Yang (Math Probl Eng 2014) | **[Hindawi]** | B+RR ("normalized RR") | 81.4 | 31.6 | 45.5 (c) | 86.2 | 73.7 | 79.5 (c) | 93 | secondhand: [Farag 2023 Tbl 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183); [Garcia 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) gives 81/31, 86/73 |
| 7 | Huang et al. (BioMed Eng OnLine 2014) | | B+RR (random projections; RR-ratio rule for S) | 91.1 | 42.2 | 57.7 | 93.9 | 90.9 | 92.4 | 93.8 | secondhand: [Cheng 2020 Tbl B.1](https://arxiv.org/abs/2007.04871); [Garcia 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) |
| 8 | Garcia et al. (Sci Rep 2017) | | B+RR (temporal VCG + 64 selected features incl. pre/post/local RR) | 62.0 | 53.0 | 57.2 | 87.3 | 59.4 | 70.7 | 92.4 | [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/) |
| 9 | Teijeiro et al., abductive interpretation (IEEE JBHI 2018) | | MB (whole-record rhythm + morphology abstraction, clustering) | 87.17 | 83.98 | 85.5 (c) | 94.63 | 96.79 | 95.7 (c) | – | [arXiv 1609.05632 Tbl 1](https://arxiv.org/abs/1609.05632) (Artificial Intelligence companion paper) |
| 10 | Mondéjar-Guerra et al. (BSPC 2019), SVM ensemble | | B+RR (pre, post, local past-10, global past-5 min; rec-avg-normalised copies) + wavelet / HOS / LBP / morph | 78.1 | 49.7 | 60.7 (c) | 94.7 | 93.9 | 94.3 (c) | 94.5 | secondhand: [Farag 2023 Tbl 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183) and [Mommen 2026 Tbl 5](https://arxiv.org/abs/2609.32854); Acc / jκ 0.773 in [authors' README](https://github.com/mondejar/ecg-classification) |
| 11 | Mousavi & Afghah (ICASSP 2019) | **[conference]** | MB (sequence of beats, bidirectional-RNN seq2seq; SMOTE on DS1) | 88.94 | 92.57 | 90.7 (c) | 99.94 | 99.50 | 99.7 (c) | 99.53 | [arXiv full text](https://arxiv.org/abs/1812.07421) — **outlier, see Q4** |
| 12 | Guo, Sim & Matuszewski (2019) | **[arXiv version read]** | MB (10-s segments; DenseNet + GRU; **segment label = most frequent annotation**) | 62.70 | 61.21 | 61.94 | 91.25 | 88.30 | 89.75 | – | [arXiv 1810.04121 Tbl 4](https://arxiv.org/abs/1810.04121) |
| 13 | Niu et al., symbolic + multi-perspective CNN (IEEE JBHI 2020) | | B + rhythm (symbolic representation that "jointly represent[s] the morphology and rhythm of the heartbeat") | 76.5 | 76.6 | 76.6 | 85.7 | 94.1 | 89.7 | 96.4 | [abstract](https://pubmed.ncbi.nlm.nih.gov/31545750/) (Acc, F1); Se / +P secondhand from [Cheng 2020](https://arxiv.org/abs/2007.04871) |
| 14 | Cheng et al., end-to-end ResNet **without** RR (2020) | **[arXiv-only]** | B (1.96-s window, 2 leads) | 25.8 | 41.8 | 31.4 | 95.6 | 78.9 | 85.8 | 91.0 | [arXiv 2007.04871 Tbl B.1](https://arxiv.org/abs/2007.04871) |
| 15 | Same model **with** RR (pre + post, rec-avg-normalised) | **[arXiv-only]** | B+RR | 44.5 | 52.2 | 46.7 | 96.4 | 83.4 | 89.2 | 91.9 | same |
| 16 | Wang et al., CWT + CNN (Entropy 2021) | **[MDPI]** | B+RR (pre, post, pre/post ratio, local = mean of 10 previous; minus "average RR") | 74.56 | 89.54 | 81.37 | 95.65 | 93.25 | 94.43 | 98.74 (abstract; see conflict) | [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/) |
| 17 | Li, Qian & Li, improved residual CNN (CMPB 2022) | | MB (5-s overlapping segments, re-labelled; **segment-level**) | 35.22 | 65.88 | 45.9 (c) | 88.35 | 79.86 | 83.9 (c) | – | [abstract](https://pubmed.ncbi.nlm.nih.gov/34933228/) |
| 18 | Zahid, Kiranyaz & Gabbouj, 1D Self-ONN + RR injection (IEEE TBME 2023) | | B+RR ("inject temporal features based on RR interval") | 82.50 | 82.19 | 82.34 | 96.10 | 94.41 | 95.2 | – | [abstract](https://pubmed.ncbi.nlm.nih.gov/35786545/) |
| 19 | Farag, matched-filter CNN model 6 (Sensors 2023) | **[MDPI]** | B+RR (pre and post divided by past-80 / past-400 RR means) | 81.60 | 82.68 | 82.14 | 95.00 | 95.63 | 95.31 | 98.18 | [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183) |
| 20 | Farag 2023, RR-only XGBoost (no morphology) | **[MDPI]** | RR only | 60 | 36 | 45 | 58 | 75 | 65 | 93 | same |
| 21 | Yan, Zhou & Wong, SNN (BSPC 2021) | | B (180 samples, SMOTE on training data) | 67.0 | 46.0 | 54.5 | 77.0 | 59.0 | 66.8 | 90.0 | secondhand: [Mommen 2026 Tbl 4–5](https://arxiv.org/abs/2609.32854) **[arXiv-only source]**. **Conflict:** [Silva 2025](https://arxiv.org/abs/2503.07276) describes what appears to be the same two-stage CNN→SNN work (Shenjing accelerator, SMOTE, 180-point windows) as reaching "sensitivities of 91% for SVEB and 85% for VEB" with its four-class stage. That figure is probably second-stage-only (beats already flagged abnormal) and not end-to-end; unverified. |
| 22 | Kachuee CNN re-trained DS1 → DS2 (re-implementation) | **[arXiv-only]** | B (187-sample beat) | 0.0 | 0.0 | 0 | 92.4 | 30.9 | 46.3 (c) | 81.2 | [Cao et al. 2206.14200 Tbl 2](https://arxiv.org/abs/2206.14200) |

Notes on the table:
- **Row 9 (Teijeiro):** the same table also gives Teijeiro on DS1+DS2: V 92.82/92.23, S 85.10/84.51. It lists assisted methods separately; see Q4. [arXiv 1609.05632](https://arxiv.org/abs/1609.05632) **[arXiv version of a journal paper; the numbers were copied from its Table 1]**
- **Row 12 (Guo):** in the same paper's re-implementation, Rahhal et al. score S 37.8/40.5 and Jiang et al. S 50.6/67.9. Guo excluded methods that "have used a good number of annotated beats from the testing records for training purposes". Training and test sets were "standardised with a mean of 0 and a standard deviation of 1 separately". [arXiv 1810.04121](https://arxiv.org/abs/1810.04121)
- **Row 16 (Wang 2021) inconsistency:**
  - The F class is almost never detected: Se 0.26, +P 2.04.
  - The overall accuracy is internally inconsistent: the abstract says 98.74 %, while Table 3 gives 98.74 for S and 99.27 for V.
  - The confusion matrix sums to about 97.48 % (48,411 / 49,661).
  - Source: [PMC7831114](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/)
- **Row 19 (Farag 2023):** model 1 of the same paper reaches S Se 85.30 / +P 58.06 / F1 69.09 and V 96.34 / 90.20 / 93.17, with accuracy 96.48. The abstract's S Se 85.3 comes from model 1, not model 6. [PMC9919183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)
- **Assisted / patient-adapted results are not comparable with automatic ones.** On DS2 they score ([Teijeiro Tbl 1](https://arxiv.org/abs/1609.05632)):
  - de Chazal & Reilly 2006 (assisted): V 93.4/97.0, S 94.0/62.5.
  - Llamedo & Martínez (assisted): V 93/97, S 92/90.
  - Kiranyaz et al. (assisted): V 95.0/89.5, S 64.6/62.1.
  - The de Chazal & Reilly 2006 abstract: "An expert then validates and if necessary corrects a fraction of the beats of the recording. The system then adapts …" [PubMed 17153211](https://pubmed.ncbi.nlm.nih.gov/17153211/)
- **Other SNN result (secondhand):** Mao et al., IEEE TBioCAS 2022 (SNN processor) reports inter-patient Acc 93.70 only, with no per-class values in that table. [Mommen 2026 Tbl 4](https://arxiv.org/abs/2609.32854)
- **Mommen et al. 2026 (logic-gate networks, arXiv-only):**
  - Best: Acc 94.41, jκ 0.683; LGN^r S +P 74.5 / Se 33.8 / F1 46.5.
  - These models use the next R-peak (R_p1) in their RR features.
  - Their Table 5 rows for "Deep CNN [Li 2022]" and "SVM ensemble" look column-shifted. The text says the deep CNN has the highest V F1 of 0.943, but the table assigns 0.943 to the SVM ensemble. Use that table with caution.
  - Source: [arXiv 2609.32854](https://arxiv.org/abs/2609.32854)
- **Spread of earlier deep results:** Cheng et al. 2020 summarised prior inter-patient deep / feature work as "89–96% overall accuracies, 23–77 F1 for SVEB, and 63–90 F1 for VEB". [arXiv 2007.04871](https://arxiv.org/abs/2007.04871)
- **Other summary metric:** Mondéjar-Guerra's README compares accuracy / mean sensitivity / jκ:
  - Ensemble of SVMs: 0.945 / 0.703 / 0.773
  - Single SVM: 0.884 / 0.696 / 0.640
  - Zhang 2014: 0.883 / 0.868 / 0.663
  - Mar 2011: 0.899 / 0.802 / 0.649
  - de Chazal 2004: 0.862 / 0.832 / 0.612
  - Source: [github.com/mondejar/ecg-classification](https://github.com/mondejar/ecg-classification)
- **Which methods use RR or neighbouring beats:** rows 1–10, 13, 15, 16 and 18–20 use RR explicitly. Rows 9, 11, 12 and 17 use multi-beat context. Rows 14, 21 and 22 are single-beat only, and they have the lowest S F1 (0–54.5).

### Inferences
- **Realistic bars for a new method claiming "≈ state of the art" under the strict protocol:**
  - S F1 ≥ ~76 (Niu 2020) to ~82 (Zahid 2023, Farag 2023).
  - V F1 ≥ ~90–95.
  - Classical baselines (S F1 ~50–61) are easy to beat.
- **Two-axis comparison:** the S +P axis is where classical methods are weakest. They over-call S in N-dominant patients: de Chazal has 1,904 N→S errors. Results should be compared on both Se and +P (or F1), not Se alone.
- **Not comparable with beat-level DS2 results:** segment-level results (Guo 2019, Li 2022) and assisted results.
- **Mousavi & Afghah 2019** (S F1 ≈ 91, V F1 ≈ 99.7, essentially no intra/inter gap) is an outlier versus every other paper. It should be cited with a caveat rather than as a target; see Q4.

### Gaps
- **Sellami & Hwang 2019** (Expert Systems with Applications; 3-beat input with batch-weighted loss) is often cited for high S Se. I could not verify its numbers from the paper or a reliable table.
- **Mondéjar-Guerra 2019's own S/V numbers and its per-feature-type SVM results** (which would give an RR-alone row) were not seen in the original. The ScienceDirect and repository copies were not reachable. The S/V numbers above come from two independent secondary tables that agree: 78.1/49.7 and 78.1/49.8; 94.7/93.9 and 94.8/93.8.
- **Mar 2011 and Lin & Yang 2014** per-class values are secondhand only. The Augsburg OA PDF of Mar 2011 timed out, and the Hindawi PDF was blocked.
- **Recent 2024–2026 deep inter-patient papers** were not verified (e.g. iScience 2024, Sci Rep 2026 hierarchical, residual-attention two-phase training reporting S F1 ≈ 0.74). Search snippets only: a Sci Rep 2026 hierarchical framework reports S Se 93.23 %, but its +P was not seen.

---

## Q4. Standard pitfalls that reviewers check, and who raised them

### Takeaway
The recognised pitfalls are:
1. Random beat / intra-patient splits (de Chazal 2004; Luz & Menotti 2011).
2. Using test-record labels for adaptation or tuning, i.e. "assisted" or patient-specific methods (named in de Chazal 2004 for Hu et al.; separated by Teijeiro and by Guo).
3. Resampling, augmenting or balancing the test set (Cao et al. on Kachuee; Mousavi & Afghah on Acharya 2017).
4. Using DS2 for model selection (de Chazal's explicit rule).
5. Records 201/202 come from the same subject but sit in different partitions (PhysioNet; Silva et al. 2025).
6. Non-causal RR features: post-RR and whole-record averages. These are well known but rarely treated as an error. The literature mostly treats them as acceptable for offline Holter analysis. I found no paper that quantifies their effect.

### Cited Findings
- **Beat-wise splitting:** de Chazal 2004 (quoted in Q2); [Luz & Menotti 2011](https://pubmed.ncbi.nlm.nih.gov/22255458/); [Garcia 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/); [Cao et al.](https://arxiv.org/abs/2206.14200); [Silva et al. 2025](https://arxiv.org/abs/2503.07276).
  - A further check found by Cao et al.: Romdhane et al. "claimed that their model was evaluated using inter-patient split paradigm even if it is based on intra-patient split" ([arXiv 2206.14200](https://arxiv.org/abs/2206.14200)).
- **Patient-specific / assisted use of test-patient data:**
  - de Chazal 2004 contrasts its automatic system with Hu et al.'s mixture-of-experts, whose "local classifier requires a cardiologist to annotate a segment of a patient-specific ECG". The global classifier alone reached 62.2 % and the expert-mixture 94.0 % VEB accuracy. [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)
  - Teijeiro's comparison table separates "Automatic" from "Assisted" methods. [arXiv 1609.05632](https://arxiv.org/abs/1609.05632)
  - Guo et al. exclude works "that have used a good number of annotated beats from the testing records for training … they do help the model learn patient or even record specific patterns". [arXiv 1810.04121](https://arxiv.org/abs/1810.04121)
  - Merdjanovska & Rashkovska call patient-specific evaluation a separate setting that permits "including limited ECG segments from the patient in question in the training set". [PMC10356811](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356811)
  - Ye, Kumar & Coimbra 2016 is a related variant: it uses "a small amount of unlabeled personal data" from the test subject for adaptation and reaches 99.4 % V and 98.3 % S accuracy. [PubMed 26285228](https://pubmed.ncbi.nlm.nih.gov/26285228/) **[abstract only]**
- **Test-set resampling and augmentation:**
  - Cao et al. on Kachuee 2018: imbalance mitigation "even in the 'test set' which is contrary to machine learning practice". [arXiv 2206.14200](https://arxiv.org/abs/2206.14200)
  - Mousavi & Afghah: "the work done by Acharya et al. [18] used artificially augmented dataset to build the model, and their model evaluation was performed using augmented data". They state that they applied SMOTE "after splitting … only performed on the training dataset … (and on DS1 for the inter-patient paradigm)". [arXiv 1812.07421](https://arxiv.org/abs/1812.07421)
- **DS2 kept out of development:** "This assessment is unbiased as DS2 was not used at any point in the development of the classifier." [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf). A paper that early-stops or tunes on DS2 breaks this.
- **Same-subject records across partitions:** "For subject-independent evaluation, records 201 and 202 should be assigned to the same partition because they originate from the same subject". [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). [Silva et al. 2025](https://arxiv.org/abs/2503.07276) notes that the standard split violates this.
- **Future information inside RR features:**
  - de Chazal's own definitions use post-RR, a whole-record average and a centred 10-interval local average (Q1).
  - **Mondéjar-Guerra's public code:**
    - `local_RR` averages the last 10 pre-RR values (past only).
    - `global_RR` averages past RR within 108,000 samples (5 min, past only).
    - `post_RR` is the next interval.
    - The "normalized RR" features divide by `np.average` over the whole record: `avg_pre_R = np.average(RR[p].pre_R)` in `load_MITBIH.py`.
    - Sources: [features_ECG.py](https://github.com/mondejar/ecg-classification/blob/master/python/features_ECG.py); [load_MITBIH.py](https://github.com/mondejar/ecg-classification/blob/master/python/load_MITBIH.py)
  - Cheng et al. normalise pre-/post-RR "by the average RR interval over each recording". [arXiv 2007.04871](https://arxiv.org/abs/2007.04871)
  - Wang et al. subtract "the average RR-interval" (window unspecified) and use post-RR. [PMC7831114](https://pmc.ncbi.nlm.nih.gov/articles/PMC7831114/)
  - Farag deliberately uses only past windows for the means (last 80 / last 400 RR) but still uses post-RR. [PMC9919183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)
  - The one-beat dependency is visible in pipelines that must drop the last beat because it has "no Post-RR interval" ([AFD paper, arXiv 1906.07361](https://arxiv.org/abs/1906.07361) — seen only in a search snippet).
- **Bidirectional or multi-beat models over beat sequences** (Mousavi & Afghah's BiRNN encoder over a "sequence of ECG beats") see future beats by construction. Mousavi also resizes each T-to-T segmented beat to 280 samples. [arXiv 1812.07421](https://arxiv.org/abs/1812.07421)
- **Test-set statistics in normalisation:** Guo et al. standardise "training and testing datasets … separately", so test-set statistics are used. [arXiv 1810.04121](https://arxiv.org/abs/1810.04121). I found no paper that names this as a pitfall; it is my observation.
- **Metric conventions that can mislead:**
  - AAMI VEB statistics ignore F/Q predicted as V ([de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)).
  - Overall accuracy is dominated by N. Garcia et al. stress Se and +P for this reason ([PMC5585360](https://pmc.ncbi.nlm.nih.gov/articles/PMC5585360/)).
  - Many papers drop the Q class (7 DS2 beats) or redefine S (e.g. adding e and j beats; [Mommen 2026](https://arxiv.org/abs/2609.32854)). Beat counts therefore differ across papers.
- **Annotated R-peaks:** nearly all papers use the database's manually corrected beat positions. de Chazal notes that automatic detection would add errors to the RR features. [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)

### Inferences
- **Checklist a reviewer would apply to a new method:**
  1. DS1 only for training and model selection (no DS2 early stopping).
  2. No test-set resampling or augmentation.
  3. No test-record labels.
  4. State whether any test-record unlabeled statistics are used: record-mean RR, test standardisation, domain adaptation.
  5. State the latency / causality of the context: post-RR, centred windows, bidirectional context.
  6. Report S and V Se / +P / F1 with the AAMI conventions and the exact beat counts.
  7. Optionally report a variant that keeps 201 and 202 together.
- **Mousavi & Afghah 2019's near-perfect inter-patient numbers** are best treated as unreliable. They are far above every other same-protocol result, their intra and inter results are almost equal, and a bidirectional multi-beat context is combined with SMOTE on sequences. No explicit published refutation was found, so this is an inference.

### Gaps
- I found no published re-analysis that quantifies the effect of the 201/202 leak, of record-mean RR normalisation, or of test-set standardisation on DS2 metrics.
- The Luz 2016 survey's list of pitfalls was not seen.

---

## Q5. Is there an accepted causal (past-only, real-time) variant, and how much does it lose?

### Takeaway
**No accepted, standardised causal variant of the de Chazal protocol was found.** Almost all inter-patient papers, including "real-time" and edge-hardware papers, still use post-RR, which costs one beat of latency. Several replace the whole-record average with a past-window average:
- Mondéjar-Guerra code: past 10 beats and past 5 min.
- Farag 2023: past 80 / 400 intervals.
- Mommen 2026: past 500 intervals.

**No paper found reports DS2 metrics for "past-only" vs "with post-RR / with record average" on the same model,** so the cost of strict causality is unquantified. The closest evidence is about RR as a whole:
- Adding pre+post RR (record-normalised) to a single-beat deep model raised S F1 from 31.4 to 46.7 (Cheng 2020).
- RR alone, with past-normalised pre and post, gives S F1 45 and V F1 65 (Farag 2023).

### Cited Findings
- **de Chazal 2004 defines post-RR as "the RR-interval between a given heartbeat and the following heartbeat"** and the average RR over the whole recording. So the reference protocol is offline. [de Chazal 2004](http://web.archive.org/web/20180920170223/http://www.tara.tcd.ie:80/bitstream/handle/2262/17623/automatic.pdf)
- **Farag 2023:**
  - Local mean = "moving average operator to the last 80 RR intervals". Global mean = "averaging the last 400 RR intervals". Pre- and post-RR are divided by these means.
  - The means are causal, but post-RR is still a feature.
  - The RR-only XGBoost (used to choose the windows) gives S recall 60 / precision 36 / F1 45 and V 58 / 75 / 65, with accuracy 93.
  - Source: [PMC9919183](https://pmc.ncbi.nlm.nih.gov/articles/PMC9919183)
  - [Silva et al. 2025](https://arxiv.org/abs/2503.07276) describe these as "dynamic normalized RR intervals … calculated using a moving average over past intervals".
- **Mondéjar-Guerra code:** past-only local and global RR (the comment reads "TODO: AVG from past 5 minutes = 108000 samples"), but post-RR plus whole-record-mean normalised copies. [features_ECG.py](https://github.com/mondejar/ecg-classification/blob/master/python/features_ECG.py); [load_MITBIH.py](https://github.com/mondejar/ecg-classification/blob/master/python/load_MITBIH.py)
- **Mommen et al. 2026 (FPGA / logic-gate networks):** RR features come from three previous R-peaks and the next R-peak (R_p1). The local RR coefficient of variation uses the 500 previous intervals and "was updated for each heartbeat". This is again past-normalised but still needs one future beat. [arXiv 2609.32854](https://arxiv.org/abs/2609.32854) **[arXiv-only]**
- **Cheng et al. 2020 RR ablation** (same ResNet; pre+post RR normalised by recording average):
  - S Se 25.8 → 44.5, +P 41.8 → 52.2, F1 31.4 → 46.7.
  - V F1 85.8 → 89.2.
  - Overall accuracy 91.0 → 91.9.
  - Source: [arXiv 2007.04871](https://arxiv.org/abs/2007.04871) **[arXiv-only]**
- **Why rhythm matters most for S:** Teijeiro et al. say S beats "are usually hard to distinguish using only morphological features", and that their "rhythm description of signal" enables the improvement. [arXiv 1609.05632](https://arxiv.org/abs/1609.05632)
- **Multi-beat windows help on MIT-BIH:** among segmentations, "beats2.4s performs the best overall in terms of macro F1". That segment is "a heartbeat and its two neighboring ones", so it is centred and non-causal. [Merdjanovska & Rashkovska 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356811)

### Inferences
- **A strictly causal variant would drop several cues:**
  - post-RR, i.e. the compensatory pause after a premature beat (inference from physiology, not from a measured ablation);
  - record-mean normalisation;
  - centred local averages;
  - bidirectional context.
- **Expected effect, by inference only:** most of the loss should fall on S beats, because S detection relies mostly on timing relative to the patient's own rhythm.
- **One-beat-delay variant:** a design that is causal except for one beat of latency (past-window normalisation + pre-RR + post-RR, as in Farag 2023) is the closest thing to an accepted "real-time" practice. It reaches S F1 ≈ 82 and V F1 ≈ 95 on DS2.
- **A fully past-only model (no post-RR):** I found no published DS2 number for one, so its cost would need to be measured directly.
- **Practical recommendation for a new method that claims to use beat context:** report three versions. They are (a) causal only, (b) one-beat delay with post-RR, and (c) full / centred context. The causal variant should use past-window normalisation, not record-mean normalisation.

### Gaps
- No same-model DS1 → DS2 ablation removing post-RR alone, or replacing the record average with a past-window average, was found in any venue.
- I did not examine Llamedo & Martínez 2011's exact 8 selected features. Their abstract only says "features from the RR series". It may show whether post-RR was selected.
