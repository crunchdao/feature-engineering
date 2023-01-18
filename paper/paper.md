---
title: "Machine Learning meets Statistical Physics: a Web3 perspective"
subtitle: "CrunchDAO"
author: [Matteo Manzi, Utkarsh Pratiush, Enzo Caceres]
date: "2023/01/18"
lang: "en"
colorlinks: true
titlepage: true
titlepage-text-color: "FFFFFF"
titlepage-rule-color: "360049"
titlepage-rule-height: 0
titlepage-background: "./figures/cover.pdf"
header-left: "\\hspace{1cm}"
header-right: "Page \\thepage"
footer-left: "Machine Learning meets Statistical Physics: a Web3 perspective"
footer-right: "CrunchDAO"
abstract: "CrunchDAO's Machine-Learning-enabled ensemble framework builds on top of traditional econometric risk models, requiring a number of steps in the data preparation: features orthogonalization, standardization, model order reduction and data obfuscation will be discussed. It is discussed how, in the context of ensemble learning and bagging in particular, combining a variety of orthogonal models yields more accurate estimates of expectations. Moreover, the statistics of the set of predictions can be used to infer a measure of risk in the portfolio management process. We discuss how to integrate this in modern portfolio theory. We briefly discuss the necessary relation between these design choices and the ergodic hypothesis on financial data.
"
---

# Econometrics

In a multi-factor framework of $M$ factors ([@Sharpe_1964], the total excess return over the risk-free rate of asset $i \in [1, ..., N]$ is given by

\begin{equation}
    r_i = \sum_{j = 1}^M \beta_{ij} F_j + \epsilon_i
\end{equation}

with $M \ll N$. $\beta_{ij}$ identifies entries of the $\textbf{B}$ matrix, $F_j$ of the \textbf{F} vector and $\epsilon_i$ of the \textbf{E} vector.

# References
