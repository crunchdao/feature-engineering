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

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_0.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_1.png}
\endminipage\hfill
\minipage{0.33\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_2.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_3.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_4.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_5.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_6.png}
\endminipage
\end{figure}

# Orthogonalization

One reason for which predictions $\hat{Y}$ with good linear correlation 

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

in which the last implication is due to the fact that, in a portfolio management framework, $\omega$ and $\hat{Y}$ are naturally highly correlated.

\begin{figure}[!htb]
  \includegraphics[width=.3\linewidth]{plots/exposure.png}
\end{figure}

\begin{figure}[!htb]
  \includegraphics[width=.25\linewidth]{plots/gram.png}
\end{figure}

We cannot use Gram-Schmidt (iteratively orthogonalize against all factors), as factors in general do not define an orthogonal basis. We project each feature in a least-square sense:

\begin{equation}
    h(X) = X - B \left( B^T B\right)^{-1} B^T X
\end{equation}

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/exposure_O.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/covariance_O.png}
\endminipage
\end{figure}


\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_0_O.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_1_O.png}
\endminipage\hfill
\minipage{0.33\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_2_O.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_3_O.png}
\endminipage\hfill
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_4_O.png}
\endminipage\hfill
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_5_O.png}
\endminipage
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_6_O.png}
\endminipage
\end{figure}

# Gaussianization

\begin{figure}[!htb]
  \includegraphics[width=.7\linewidth]{plots/schematic_lambert.png}
\end{figure}

At this point features are strongly non-Gaussian [@taleb2020], while the volatility of the first four statistical moments is small enough for us to define an invariant measure to Gaussianize them [@Goerg_2010], [@Arbabi_2019], [@Marti_2016]. This step also further reduces the non-stationarity of the features.

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_0_OG_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_1_OG_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_2_OG_gaussian.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_3_OG_gaussian.png}
\endminipage\hfill
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_4_OG_gaussian.png}
\endminipage\hfill
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_5_OG_gaussian.png}
\endminipage
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_6_OG_gaussian.png}
\endminipage
\end{figure}

As the kernel implicitly defined by the Gaussianization step is nonlinear, the orthogonality condition is destroyed: we can however perform again orthogonalization and obtain orthogonal, Gaussian features.

Moreover, performing the three steps OGO, compared to only the first O step, leads to features which are always more than 98.09\% Spearman rank correlated.

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_0_OGO_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_1_OGO_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_2_OGO_gaussian.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_3_OGO_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_4_OGO_gaussian.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_5_OGO_gaussian.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_6_OGO_gaussian.png}
\endminipage
\end{figure}

# Standardization

Given the degree of stationarity, we can standardize using a global transformation.

# Principal Component Analysis

We perform Principal Component Analysis (PCA) to decorrelate the features as much as possible. Again, we perform a global basis transformation thanks to the fact that features are close to stationary.

The Gaussianization step is implicitly introducing a kernel, so that this procedure can be though of a specific case of Kernel PCA [@nateghi2023].

We here obtain new features linearly combining them. The linear combination coefficients come from Single Value Decomposition: we can hence standardize again.

\begin{equation}
X_j^{out} =\frac{1}{\sigma_{j2}}\beta_j \sum_i \alpha_{ij} \frac{1}{\sigma_{j1}} \cdot \left( k_j\left( X_i^{in} - A_{i1} \right) - A_{i2} \right)
\end{equation}

All the steps after the second orthogonalization are linear transformations: output features remain orthogonal to the factor space.

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_0_final.png}
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_1_final.png}
\endminipage\hfill
\minipage{0.33\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_2_final.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_3_final.png}
\endminipage\hfill
\minipage{0.25\textwidth}
  \includegraphics[width=\linewidth]{plots/feature_4_final.png}
\endminipage\hfill
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_5_final.png}
\endminipage
\minipage{0.25\textwidth}%
  \includegraphics[width=\linewidth]{plots/feature_6_final.png}
\endminipage
\end{figure}

\begin{figure}[!htb]
  \includegraphics[width=.55\linewidth]{plots/final_correlation.png}
\end{figure}

# Quantization

We perform Lloyd-Max Quantization [@Lloyd_1982] to solve a classification problem in a least-square sense.

\begin{figure}[!htb]
  \includegraphics[width=0.4\linewidth]{plots/feature_0_quantized.png}
\end{figure}
\begin{figure}[!htb]
  \includegraphics[width=0.4\linewidth]{plots/feature_1_quantized.png}
\end{figure}

# Target

\begin{figure}[!htb]
  \includegraphics[width=.3\linewidth]{plots/target_raw.png}
\end{figure}

Targets are not Gaussian: given the knowledge of assets ranking, the distribution of expected targets is fat tailed.

\begin{figure}[!htb]
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/alpha_score.png}
\endminipage\hfill
\minipage{0.33\textwidth}
$$
\Longrightarrow
$$
\endminipage\hfill
\minipage{0.33\textwidth}
  \includegraphics[width=\linewidth]{plots/fat_alpha_score.png}
\endminipage
\end{figure}
\end{frame}

## Quantization

We constrain the distribution of the quantized targets to match the non-zero kurtosis of the target. The relative size of each bin is again obtained maximizing the explained variance of the classification step, for a given number of bins, like for the features.

\begin{figure}[!htb]
  \includegraphics[width=.6\linewidth]{plots/target_quantized.png}
\end{figure}

# References
