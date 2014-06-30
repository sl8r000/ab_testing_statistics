---
layout: default
title: Home
---

# The Statistics of A/B Testing

Good statistical practices for A/B testing are not especially complicated, but they are a little more complicated then they are often thought to be. This is unfortunate, because a shaky statistical foundation can lead to a flawed decision process and thus a long procession of A/B tests that "looked good" but are actually worthless or actively harmful. Below is some imperatively-phrased advice on avoiding some common pitfalls in the statistics of A/B testing. Most of this is Bayesian, primarily because I like having posterior distributions; there are some situations in which Frequentist methods are equally useful, so I'll mention them as well.

### [Use the Beta Distribution]({{ site.baseurl }}{{ site.link_beta }})
The Beta distribution is preferable to its normal approximation when modeling a \\(k\\)-successes-out-of-\\(n\\)-trials test, as one would be doing for any converstion-rate-type A/B test. Hopefully your sample size is large enough for this not to matter, but it's worth being aware of -- especially when you have a small sample, or your conversion rate is close to \\(0\\) or \\(1\\), since these are situations in which the normal approximation is particularly bad.

### [Use a Hierarchical Model]({{ site.baseurl }}{{ site.link_hier }})
Failing to correct for multiple comparisons is one of the most common A/B testing mistakes. Hierarchical models are one way to address this problem; I'll also mention some Frequentist alternatives, which are also very useful.


### [Avoid Biased Stopping Times]({{ site.baseurl }}{{ site.link_biased }})
If you stop an A/B test as soon as the results "look significant", you put yourself at a shockingly high risk of false positives. Set a stopping time in advance, or use a dynamic p-value instead.

### [Retest and Watch your Base Rate]({{ site.baseurl }}{{ site.link_retest }})
Follow-up testing is a cheap way to double-check your test. Do it! Also, be mindful of your overall success rate; when this is low, you should be especially suspicious of successful tests.

### [For God's Sake, Pay Attention to the Time Series]({{ site.baseurl }}\#)
Coming soon.