# AbGraftBERT

## Introduction

<p align="center"><img src="AbGraftBERT/figures/graft.pdf" width=80%></p>
<p align="center"><b>Schematic illustration of the ABGNN framework</b></p>


The AbBERT is the pre-trained antibody model. Its `soft' prediction will be fed into the sequence GNN $\cal{H}$<sub>seq</sub>, after encoding and generating the updated sequence, structure GNN $\cal{H}$<sub>str</sub> encodes the updated graph and then predict the structures. The sequence and structure prediction iteratively refine $T$ times.
