---
title: "Building a Risk Analysis Tool with Public Data: Behind the Scenes"
layout: post
blog: true
author: Olga Kahn
projects: true
category: project
summary:
permalink: projects/risksleuth
comments: true
---

# **Building a Risk Analysis Tool with Public Data: Behind the Scenes**

If you’re into investing or just curious about how financial tools work behind the curtain, this post is for you. Over the last few weeks, we’ve been diving deep into the process of developing a **risk analysis tool** designed to help investors understand the hidden connections and vulnerabilities within their portfolios. Think of it as finding out not just *what could go wrong* with an investment, but also how risks might ripple through a portfolio like dominoes.

## **The Problem We're Solving**

Investors face a tough challenge: **risks are rarely isolated**. A cybersecurity issue at one company could signal vulnerabilities across the industry. Regulatory crackdowns in one country could send shockwaves through global markets. Yet, most risk tools don’t focus on these connections—they only look at risks one company at a time.

We wanted to go deeper. What if we could not only flag a company’s own risks but also show how its problems might transfer to other companies in a portfolio? **That’s the heart of what we’re building: a tool that connects the dots.**

---

## **Breaking Down the Risk Metrics**

To keep things transparent, here are the core metrics we’ve developed to analyze risks. You don’t need to be a finance pro to understand them—they’re pretty intuitive:

### 1. **Inherent Risks**
These are risks that are baked into a company’s operations. For example, if a company relies heavily on a volatile market or cutting-edge tech, that’s an inherent risk. We assess these by digging into the company’s public filings (like 10-K reports) and scoring each risk based on **impact** and **likelihood**.

- *Example*: Zoom might list risks around keeping up with cybersecurity standards in its own tech infrastructure.

### 2. **Generic Risks**
These are the risks that don’t belong to any one company but are **common across the industry or market**. Imagine you’re investing in two tech companies—they likely face similar threats from data privacy regulations. We assess these risks by looking at overlaps between companies.

- *Example*: Meta’s challenges with data privacy laws could also apply generically to IBM, given shared regulatory environments.

### 3. **Transferred Risks**
Here’s where things get interesting. **Transferred risks** look at how vulnerabilities in one company could spill over to another. For example, if one company suffers a high-profile data breach, it could make customers or regulators scrutinize similar companies in the portfolio.

- *Example*: IBM’s vulnerabilities in cybersecurity could increase risk for Zoom, given their shared dependence on secure client data systems.

### 4. **Differential Severity**
This measures whether a risk becomes **more severe** or **less severe** when transferred between companies. If a risk gets amplified in the transfer, it’s a warning sign of interconnected vulnerabilities.

- *Example*: If Zoom faces greater cybersecurity risks when considering IBM’s profile, we’d flag that as a heightened concern.

### 5. **Weighted Severity**
This combines all the above into a single score. It’s like a composite risk grade for each company, factoring in its own risks (inherent), industry-wide risks (generic), and connections to others (transferred).

---

## **From Metrics to Insights**

The beauty of building this tool is seeing how these metrics come together to uncover patterns you wouldn’t spot otherwise. Here are a few real-life insights we’ve drawn from testing the tool:

### **1. Meta Is Carrying Industry Risks on Its Shoulders**
- Meta’s weighted severity score is the highest in the portfolio. This is partly because of its **high transferred severity**, meaning it’s deeply affected by risks originating from other companies.
- *Takeaway for investors*: Meta might be a high-reward, high-risk investment. It’s essential to monitor not just its own moves but broader industry trends (e.g., regulatory crackdowns or cybersecurity developments).

### **2. Accenture Offers a Safer Bet**
- Accenture’s risk profile is much calmer. Its weighted severity is the lowest, driven by lower inherent and transferred risks.
- *Takeaway*: Accenture could act as a stabilizing force in a tech-heavy portfolio, offsetting higher-risk companies like Meta or Zoom.

### **3. Cybersecurity: The Common Thread**
- Cybersecurity risks popped up across the board, often with **high transferred severities**. This suggests that vulnerabilities at one company (e.g., IBM) could signal broader concerns for others (e.g., Zoom).
- *Takeaway*: If your portfolio leans heavily on tech, cybersecurity should be a priority—not just for one company but across the sector.

---

## **Making It Useful for Investors**

We didn’t just want to throw charts and scores at people. A risk analysis tool is only valuable if it leads to **clear, actionable recommendations**. Here’s how we’re thinking about that:

### **Portfolio Rebalancing**
- High-risk companies like Meta might need reduced exposure to avoid volatility.
- Low-risk companies like Accenture could take up a larger share to provide balance.

### **Diversification**
- High **generic severity** scores across the portfolio suggest a need for sector diversification. For example, adding companies from industries like healthcare or utilities could reduce reliance on tech-heavy investments.

### **Proactive Risk Management**
- High **transferred severities** in cybersecurity mean investors should keep an eye on how interconnected risks evolve. This could mean following regulatory changes, company updates, or broader industry trends.

---

## **What’s Next for the Tool?**

Building in public means being open about what’s working and what’s still a work-in-progress. Here’s what we’re tackling next:

1. **Adding Real-Time Updates**
   - Risks evolve quickly, and we want the tool to reflect changes as they happen. For example, a new data breach at IBM should immediately adjust scores for other companies in the portfolio.

2. **Visualization Tools**
   - Let’s be honest—no one wants to stare at rows of numbers. We’re working on intuitive graphs and charts to make it easier to spot patterns and correlations.

3. **Investor Personalization**
   - Not all investors have the same goals. Some might tolerate higher risks for bigger returns, while others prioritize stability. We’re adding filters and custom settings to tailor insights to individual strategies.

---

## **What You Can Learn From This Process**

Even if you’re not a techie or a finance pro, there’s something relatable about building tools like this:

- **Everything is Connected**: Whether it’s in investing, running a business, or even managing personal relationships, risks rarely exist in isolation. Seeing the connections can help you make better decisions.
- **Small Patterns, Big Insights**: The magic isn’t just in the big numbers—it’s in the subtle differences. For example, seeing how one company amplifies another’s risks tells a much richer story than just listing generic risks.
- **Iterate, Iterate, Iterate**: Our first models were clunky and overly complicated. But by testing and refining, we’re creating something investors can actually use—and trust.

---

## **Conclusion**

Developing a risk analysis tool using public data has been a fascinating challenge. It’s about more than just crunching numbers; it’s about uncovering the stories those numbers tell and making them actionable for investors.

Whether you’re a DIY investor or just curious about how tech can transform decision-making, there’s a lot to take away from this process. If nothing else, let this be a reminder that understanding risk isn’t just about avoiding it—it’s about **managing it wisely**.

What would you want to see in a tool like this? Let us know—we’d love to hear your thoughts as we continue building this in public! 
