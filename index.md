---
layout: default
title: Home
---

# Stastical Advice for A/B Testing

A/B testing is awesome. It's fun, it's lucrative, and it's one of the most visible and impactful things that you can do as a data scientist / statistician / anyone-interested-in-optimization at a company. It's undeniably satisfying to see a change you proposed make a multi-million dollar difference. (If only you could get paid on commission!)

Before you write that email to your CEO demanding a raise, though, it's worth making sure that your test-evaluating process is correct. It would be... unfortunate if subsequent testing showed that your decision to color all your CTA buttons hot pink wasn't worth the "mad stacks" that you claimed, and was in fact actively harmful. To avoid such embarassments, you'd like to implement some sound statistical practices for evaluating your A/B tests.

Unfortunately, good statistical methods for A/B testing are more complicated then they are sometimes thought to be. (If you haven't already read it, check out the whitepaper ["Most Winning A/B Test Results are Illusory"](http://www.qubitproducts.com/sites/default/files/pdf/most_winning_ab_test_results_are_illusory.pdf).) Questionable practices seem to be widespread (take a look at the first few hits on Google), and these mistakes can fatally bias your A/B testing program.

So here are five recommendations to correct for what are in my experience the most frequent difficulties. This is scattered advice for specific issues, rather than a blueprint for the elusive Perfect Process for Evaluating A/B Tests, but I hope that it's helpful in thinking about the way you'd like to run your experiments. Most of this articles are written from a Bayesian perspective (primarily because I like having posterior distributions), but there's some Frequentist content as well - both approaches have their merits in different situations.

### [Use the Beta Distribution]({{ site.baseurl }}{{ site.link_beta }})
The Beta distribution is preferable to its normal approximation when modeling a \\(k\\)-successes-out-of-\\(n\\)-IID-trials test, e.g., a converstion-rate-based A/B test. Hopefully your sample size is large enough for this not to matter, but it's worth being aware of - especially when you have a small sample, or your conversion rate is close to \\(0\\) or \\(1\\), since these are situations in which the normal approximation is particularly bad.

### [Use a Hierarchical Model]({{ site.baseurl }}{{ site.link_hier }})
Failing to correct for multiple comparisons is one of the most common A/B testing mistakes. Hierarchical models are one way to address this problem.

### [Avoid Biased Stopping Times]({{ site.baseurl }}{{ site.link_biased }})
If you stop an A/B test as soon as the results "look significant", you put yourself at a shockingly high risk of false positives. Set a stopping time in advance, or use a conservative dynamic p-value.

### [Retest and Watch your Base Rate]({{ site.baseurl }}{{ site.link_retest }})
Follow-up testing is a cheap way to double-check your test. Do it! Also, be mindful of your overall success rate; when this is low, you should be especially suspicious of successful tests.

### [For God's Sake, Pay Attention to the Time Series]({{ site.baseurl }}\#)
Coming soon. For now, just worry :-)
