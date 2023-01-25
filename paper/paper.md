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
abstract: "CrunchDAO's Machine-Learning-enabled ensemble framework builds on top of traditional econometric risk models, requiring a number of steps in the data preparation: features orthogonalization, standardization, model order reduction and data obfuscation will be discussed. It is discussed how, in the context of ensemble learning and bagging in particular, combining a variety of orthogonal models yields more accurate estimates of expectations. Moreover, the statistics of the set of predictions can be used to infer a measure of risk in the portfolio management process. We discuss how to integrate this in modern portfolio theory. We briefly discuss the necessary relation between these design choices and the ergodic hypothesis on financial data."
---

# Econometrics

In a multi-factor framework of $M$ factors [@Sharpe_1964], [@Fama_1993], the total excess return over the risk-free rate of asset $i \in [1, ..., N]$ is given by

\begin{equation}
    r_i = \sum_{j = 1}^M \beta_{ij} F_j + \epsilon_i
\end{equation}

with $M \ll N$. $\beta_{ij}$ identifies entries of the $\textbf{B}$ matrix, $F_j$ of the \textbf{F} vector and $\epsilon_i$ of the \textbf{E} vector.


# Factor Neutral Portfolio

In this context, the expected return is, with a set of positions represented by the vector $\mathbf{\omega}$:

\begin{equation}
\mathop{\mathbb{E}}(r) = \left( \mathbf{B} \omega \right)^T \mathop{\mathbb{E}}(\mathbf{F}) + \omega^T \mathop{\mathbb{E}}(\mathbf{E}) \approx \omega^T \mathop{\mathbb{E}}(\mathbf{E})
\end{equation}

\begin{equation}
 \sum_{i = 1}^N \beta_{ij} \omega_j  \approx 0 \ \ \  i = 1, ..., M
\end{equation}

Machine Learning can be used to obtain nonlinear models [@economl_2022], [@msciml], [@Prado_2019].

# Features

For each cross section, a feature Matrix \textbf{X}$_{N\times F}$ is our current representation of the state of the system.

In a supervised learning framework, associated with such feature matrix there are multiple targets, which for us represent the compound return orthogonal unexplained by the factor model. As we can work with multiple targets independently here, we will focus on one: \textbf{Y}.

# Orthogonalization

One reason for which predictions $\hat{Y}$ with good linear correlation:

\begin{equation}
corr(\hat{Y}, Y) \gg 0
\end{equation}

are not desirable is that, using Taylor expansion:

\begin{equation}
\hat{Y} = f(X) = A X + \sum_{n = 2}^\infty A_n X^n = A X + g(X)
\end{equation}

\begin{equation}
corr(X, B) \gg 0 \Rightarrow \left( f(X) \approx A X \Rightarrow  \left( \mathbf{B} \omega \right)^T \mathbf{F} \gg 0
\right)
\end{equation}

in which the last implication is due to the fact that, in a portfolio management framework, $\omega$ and $\hat{Y}$ are naturally highly correlated. We cannot use Gram-Schmidt (iteratively orthogonalize against all factors), as factors in general do not define an orthogonal basis. We project each feature in a least-square sense:

\begin{equation}
    h(X) = X - B \left( B^T B\right)^{-1} B^T X
\end{equation}


\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_0.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_1.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_2.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_3.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_4.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_5.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check1_dist_feature_6.png}
\centering
\end{figure}

# Gaussianization

At this point features are strongly non-Gaussian [@taleb2020], while the volatility of the first four statistical moments is small enough for us to define an invariant measure to Gaussianize them [@Goerg_2010], [@Arbabi_2019], [@Marti_2016]. This step also further reduces the non-stationarity of the features.

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_0.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_1.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_2.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_3.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_4.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_5.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check2_dist_feature_6.png}
\centering
\end{figure}

As the kernel implicitly defined by the Gaussianization step is nonlinear, the orthogonality condition is destroyed: we can however perform again orthogonalization and obtain orthogonal, Gaussian features.

Moreover, performing the three steps OGO, compared to only the first O step, leads to features which are always more than 98.09\% Spearman rank correlated.

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_0.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_1.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_2.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_3.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_4.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_5.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check3_dist_feature_6.png}
\centering
\end{figure}

# Standardization

Given the degree of stationarity, we can standardize using a global transformation.

# Principal Component Analysis

We perform Principal Component Analysis (PCA) to decorrelate the features as much as possible. Again, we perform a global basis transformation thanks to the fact that features are close to stationary.

The Gaussianization step is implicitly introducing a kernel, so that this procedure can be though of a specific case of Kernel PCA [@nateghi2023].

We here obtain new features linearly combining them. The linear combination coefficients come from Single Value Decomposition: we can hence standardize again.

\begin{equation}
X_j^{out} =\frac{1}{\sigma_{j2}} \sum_i \alpha_{ij} \frac{1}{\sigma_{j1}} \cdot \left( k_j\left( X_i^{in} - A_{i1} \right) - A_{i2} \right)
\end{equation}

All the steps after the second orthogonalization are linear transformations: output features remain orthogonal to the factor space.

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_0.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_1.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_2.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_3.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_4.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_5.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_dist_feature_6.png}
\centering
\end{figure}

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check4_corr_pearson.png}
\centering
\end{figure}

# Quantization

We perform Lloyd-Max Quantization [@Lloyd_1982] to solve a classification problem in a least-square sense.


\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_0.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_1.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_2.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_3.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_4.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_5.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_dist_feature_6.png}
\centering
\end{figure}

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check5_corr_pearson.png}
\centering
\end{figure}

# Targets

Targets are simply quantized. This is done maximizing the explained variance of the quantization scheme assuming a median distribution across all historical observations.

\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check_tg_dist_target_w.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check_tg_dist_target_r.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check_tg_dist_target_g.png}
\centering
\end{figure}
\begin{figure}[H]
\includegraphics[width=0.9\textwidth]{figures/check_tg_dist_target_b.png}
\centering
\end{figure}

# References
