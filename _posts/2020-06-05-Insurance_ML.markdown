---
layout: post
title:  "Using Gradient Boosting Machines to Model Risk Premiums"
subtitle: "a comparative study of risk premium models"
date:   2020-06-05 21:49:05 +0200
cover-img: https://samueltober.github.io/my-blog/assets/images/random.png
thumbnail-img: https://samueltober.github.io/my-blog/assets/images/tree.png
categories: Projects
usemathjax: true
---

This is a project I did for a swedish insurance company, with the goal to make their pricing strategy more competitive using machine learning. As the insurance industry is highly data driven it is no surprise that machine learning (ML) has made its way into the industry. While GLMs are still the comfort zone of most actuaries, we have in recent years seen a surge in machine learning algorithms. This study puts focus on developing and evaluating three tree-based machine learning models, starting from simple decision trees and working up to the more advanced ensemble methods random forests and gradient boosting machines.

## Insurance Pricing Fundamentals
First, it is important to understand why setting a premium that rightfully corresponds to the risk of the customer is so crucial for the insurer. We will highlight this with a simple example. Assume two insurers, A and B, exist and A has a low premium relative to the risk of loss, while B has an adequate premium in relation to the risk. In this scenario, a high risk customers would opt for A since their premium is relatively low compared to B, thus A would attract high risk customers and in effect see their margins being eaten up. On the contrary, if A's premiums are too high they would not attract any profitable customers and still lose money. In the light of this simple example, we see why a competitive pricing strategy is paramount. Perhaps, the most common pricing strategy is the frequency-severity strategy. In the frequency-severity model we assume that two factors will influence the risk of a customer:

 * Frequency (F) - number of claims per exposure time
 * Severity (S) - average loss per claim

where exposure is the time for which a risk is insured. Now, assume that an insurer has a total loss of $L$ spread out over $N$ claims and an exposure $e$. Then the effective, or technical premium would be:
$$\begin{equation}
    \tau = \mathbb{E}\left(\frac{L}{e}\right) = \mathbb{E}\left(\frac{L}{N} \lvert N >0\right) \times \mathbb{E}\left(\frac{N}{e}\right) = \mathbb{E}\left(S\right) \times \mathbb{E}\left(F\right)
\end{equation}$$
