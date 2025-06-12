# Meeting Notes

## Meeting with Ronald (11.06.2025)

### Need Introduction...
- Public transport plays an important role especially in large cities (TAXIS) BUT...
- Relevance of the domain
  - Research shows taxis don't take most efficient routes
  - Different reasons
- Describe problem
- AI can help!
- Describe the context
- One problem with existing approaches is...
- In this study I will...

In the literature review I need to tackle all of the above!

Why? How is it related...

**Write an intro!**

Don't repeat context and problem in literature review.

### Introduction Structure

1. Context & motivation: taxis are important
2. Problems! Take wrong routes
   1. Reasons... we focus on one reason: malicious reasons
3. AI can help!
   1. Examples...
   2. Problems...
   3. Differences...
   4. Privacy...
4. Therefore we want to deal with privacy by synthetic data
5. Various ways
6. We focus on linked data
   1. Because explainable

Different anomalies, anonymity, why synthetic data, what is the advantage of using linked data and explainability.

After literature review, make one section on research scenario and environment.

Describe the process. Context: we have access to this data...

**The Data**

Data preprocessing part of methodology.

**MAKE SURE TO EXPLAIN WHY PROCESS DATA IN CERTAIN WAY**

WHY? Because I want to make a classifier, it will classify suspicious neural weird.
Step of preprocessing should be supported why every...

Put methodology before the data processing.

Data is raw data, we want to use these algorithms to create synthetic data, we need to clean the current data, find distributions (references).

### Implementation
This is the process flow and how we do it. We use this library, this package, this server specs.

### Experimental Setup

### Results

### Evaluation

### Conclusion

### Discussion

---

### Key Decisions

Can create synthetic data from normal data OR create synthetic data from semantically enriched data.

Have a good story WHY.

Enrichment without:
- Heuristic stats

**Problem:** Data has no labels

**Solution:** We add additional classes

Can generate synthetic data easily from regular data.

#### WHY?

There are various ways of creating synthetic data from tabular data BUT to create synthetic data from linked data is not explored.

A lot of linked data out there!

#### HOW?

We have a nice dataset...

Maybe skip linked data.

Just focus on the classification and synthetic data generation.

**Focus on routes!**

Creating artificial data for routes is interesting.

Make it small enough that it is interesting but not too much for master thesis.

Focus on routes are sensitive (data), but they are very different than other data.

Need to come up with method for creating synthetic route data...