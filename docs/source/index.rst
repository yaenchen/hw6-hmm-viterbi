.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?

If the model design's sliding window increased, this could reduce the specificity of the observations since we are grouping more data into one window. The hidden states would differ due to more grouping of the data when running ATAC-seq with a larger window, and I might expect the hidden states to be less accurate.

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?

Additional genomics data would need to be harmonized to match the existing ATAC-seq data. We would need to ensure that the integration of additional genomics data is standardized to the existing data, and the addition of observation/hidden states would introduce a different set of hypothesis that can be tested. I believe it would be important to evaluate the hidden states from the new data available separately, calculating their own transition and emission probabilities rather than lumping it all into one model.

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

The genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs) could then be compared to the functional characterization. We could now use the functional characterization as our new observed states, and the genomic annotations as new hidden states to observed the relationship between cCREs and CREs, and the observed phenotypes.


Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
