---
layout: default
title: Use the Beta Distribution
---

# Use the Beta Distribution

When you have a k-successes-out-of-n-trials-type test, you should use the Beta distribution to model your posterior distributions instead of using the normal approximation. More concretely: If you have test with \\(k\\\) success amongst \\(n\\) trials, your posterior distribution is \\(Beta(k+1, n-k+1)\\), and this is preferable to using \\(N(k/n, \sqrt{(k/n)(1-k/n)n^{-1}})\\).

(Note to experts: This is assuming you have no prior; we'll address this later when we talk about hierarchical models. This is also assuming that the trials are IID Bernoulli; we'll address this later when we talk about using the test's time series.)

For example, if you have a coin with an unknown bias \\(p\\) and that coin lands heads-up on 60 out of 100 flips, then \\(p \sim Beta(61, 41)\\). Or more to the point for us: If you send out an email campaign and get 150 conversions out of 10,000 emails sent, then the true conversion rate \\(p\\) for the campaign is \\(Beta(151, 9851)\\) distributed.

<figure>
<img src="https://i.imgur.com/RxOzIUY.png">
<figcaption style="font-size: 0.5em; text-align: center">Posterior distribution for the true value of \(p\) given 60 heads out of 100 flips. In the figure, \(p \sim Beta(61, 41)\)</figcaption>
</figure>

This is an uncontroversial claim; Beta is the *correct* distribution to apply in this situation, though the normal approximation has traditionally been used due to its computational convenience. (Traditionally, you would have seen 60 out of 100 heads modeled as \\(p \sim N(0.6, \sqrt{(0.6 \cdot 0.4)/100} \approx N(0.6, 0.049)\\).) For large sample experiments in which the observed ratio \\(k/n\\) is far away from \\(0\\) and \\(1\\), using the normal approximation is generally fine; it will be close to the Beta distribution in those cases. However, you can get into trouble with small samples -- especially when \\(k/n\\) is close to \\(0\\) or \\(1\\), as would be the case for very low conversion rates (e.g. only about 0.5% of visitors exposed to your ad click on it). And with the computational tools available today (when you no longer need to carry around CDF tables for every distribution you want to use), there's really no reason to prefer the normal model. So use Beta instead!

So what is the Beta distribution? It's a bit less well known than some other distributions, so let's talk about it in greater detail. We can describe Beta completely as follows: If \\(X\\) follows a \\(Beta(a, b)\\) distribution, then the probability mass function for \\(X\\) is: \\[p\_X(t) = \frac{t^{a-1}(1-t)^{b-1}}{B(a, b)}.\\] Here \\(B(a,b)\\) is the Beta function (whence the name for the distribution). Typically, this constant factor is not of concern to us, as it is normalized out in calculations; one only cares that \\(p_X(t) \propto t^{a-1}(1-t)^{b-1}\\). Here's what the distribution looks like for a few different values of \\(a\\) and \\(b\\).

![some beta distributions](http://i.imgur.com/mj059cS.png)

Now that we know what Beta distributions look like, let's return to two claims made in the second paragraph:

1. \\(p \sim Beta(k-1, n-k+1)\\) is the right distribution for the true rate \\(p\\) when you observe \\(k\\) successes out of \\(n\\) trials. (More formally: If you know that \\(X \sim Bin(p, n)\\) for a fixed n, and you observe \\(X = k\\), then \\(p \sim Beta(k-1, n-k+1)\\).) 
2. The Beta distribution and its normal approximation differ considerably when \\(n\\) is small and \\(k/n\\) is close to either \\(0\\) or \\(1\\).

Again, the consequence is that it's better to model your rate-based test with Beta than with the normal distribution.

Let me first convince you of [1.], that Beta is the right distribution to use in this situation. Mathematically, this is just Bayes rule: If we have no prior belief and we observe \\(k\\) successes out of \\(n\\) trials, then: \\[\mathbb{P}(t \mid \text{data}) \propto \mathbb{P}(\text{data} \mid t)\mathbb{P}(t) = \mathbb{P}(\text{data} \mid t).\\] In this case, where our \\(\text{data}\\) is observing \\(k\\) successes out of \\(n\\) trials, \\[\mathbb{P}(\text{data} \mid t) = \binom{n}{k} t^k(1-t)^{n-k} = \mathbb{P}(\text{Bin}(t, n) = k),\\] the probability that Binomial random variable with rate \\(t\\) and \\(n\\) trials is equal to \\(k\\). Again, the constant factor \\(\binom{n}{k}\\) is unimportant to us, so normally one would just write \\[\mathbb{P}(t \mid \text{data}) \propto t^k(1-t)^{n-k} = \mathbb{P}(\text{Beta}(k+1, n-k+1) = t),\\] and we get the Beta distribution, as expected.

As an addendum, let me also present an empirical argument: Let's generate 10 million \\(t\\) between \\(0\\) and \\(1\\), and then pull a single sample from a \\(\text{Bin}(t, 100)\\) distribution from each. Then we'll draw a histogram for those \\(t\\) such that \\(\text{Bin}(t,100) = 60\\). The claim is that this histogram will follow a \\(\text{Beta}(61, 41)\\) distribution. The idea here is to simulate 10 million experiments where we observed \\(60\\) successes out of \\(100\\) trials, and then see what the true success rate *actually* was in those experiments. Here it is:

```python
In [50]: true_rates = np.random.uniform(0, 1, size=10000000)

In [51]: obs = np.random.binomial(100, true_rates)

In [52]: obs60_indices = np.where(obs == 60)

In [53]: sns.distplot(true_rates[obs60_indices])
Out[53]: <matplotlib.axes._subplots.AxesSubplot at 0x11123dd50>
```

![empirical](http://i.imgur.com/TuobKfI.png)

One can see that the PDF for \\(Beta(61, 41)\\) is an extremely good fit for the empirical histogram.

So now we have two reasons to believe point [1.]: A short proof via Bayes rule, and the experiment above. Let's move on to [2.]: The Beta distribution and its normal approximation differ considerably when \\(n\\) is small and \\(k/n\\) is close to either \\(0\\) or \\(1\\).

This point is even easier to make. Here's a plot of a \\(Beta(3, 19)\\) distribution against a \\(N(0.1, \sqrt{(0.1 \cdot 0.9)/20}\\) distribution.

![comparison](http://i.imgur.com/XCd5oG6.png)

The distributions differ visibly. Using the normal approximation might cause one, e.g., to underestimate the probability that the true value falls between \\(0.2\\) and \\(0.3\\) in the above. (Notice also the mass left of zero(!) in the normal approximation.)

Numerically speaking, we can quantify the extent to which the two approximations differ by looking at their [variation difference](http://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures). As a function of \\(k\\) and \\(n\\):  \\[\delta(k,n) = \\]

\\[\frac{1}{2}\int_{\mathbb{R}} \left| \mathbb{P}(Beta(k+1, n-k+1) = t) - \mathbb{P}(N(k/n, \sqrt{(k/n)(1-k/n)/n}) = t)\right|\\]

\\[= \frac{1}{2}\int_{\mathbb{R}} \left| t^k(1-t)^{n-k} - \frac{\exp\left(-\frac{(t-k/n)^2}{2\frac{(k/n)(1-k/n)}{n}}\right)}{\sqrt{2\pi\frac{(k/n)(1-k/n)}{n}}}\right|.\\]

Let's calculate this on a \\(100\\) by \\(100\\) grid:

```python
In [113]: X = np.arange(1, 101)

In [114]: Y = np.arange(1, 101)

In [115]: X, Y = np.meshgrid(X, Y)

In [116]: Z = np.sum(np.abs(scipy.stats.beta(X+1, Y-X+1).pdf(t) - scipy.stats.norm(1.0*X/Y, np.sqrt(((1.0*X/Y)*(1 - 1.0*X/Y))/Y)).pdf(t)) for t in np.linspace(0, 1, 1000))

In [117]: Z = np.nan_to_num(Z)

In [118]: fig = plt.figure()

In [119]: ax = fig.gca(projection='3d')

In [120]: surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
```
![variation distance](http://i.imgur.com/lTuSHOs.png)

We can see the behavior that we expected: The error is worse when \\(k/n\\) is close to \\(0\\) or \\(1\\), and there's less error in the "middle" (i.e. \\(k/n\\) is far from \\(0\\) and \\(1\\)) when \\(n\\) is larger.

## Wrapping Up

In conclusion: Use the Beta distribution! It's more accurate, and just as easy to compute as its normal approximation. Moreover: It's simply the correct distribution to use when you're modeling a true rate after observing \\(k\\) successes out of \\(n\\) trials. (Formally: If you know that \\(X \sim Bin(p, n)\\) and you observe \\(X = k\\), then \\(p \sim Beta(k-1, n-k+1)\\).)

The situation becomes a little more complex when you'd like to model *several* such rates, as you would when you have an A/B test with several variants. We'll handle this problem in the next section.

## Links / References

Want to know more about the Beta distribution? Check out the [first chapter](http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Chapter1_Introduction.ipynb) of Cam Davidson-Pilon's awesome github-hosted book, *Bayesian Methods for Hackers*. 


