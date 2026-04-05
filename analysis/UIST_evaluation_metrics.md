# UIST Evaluation Metrics for `uist2026-for-reproducibility`

## Positioning

For a UIST paper, the strongest evaluation is not "does the text sound fluent?"
The abstract makes a stronger claim: AI may **augment human counterfactual
thinking**, and the effect may depend on **timing** and **prompting strategy**.

That means the most important outcomes are:

1. whether the AI suggestions were usable and relevant,
2. whether they expanded human thinking,
3. whether they changed the human's final plans, and
4. what design properties users wanted from the system.

Automatic text metrics can stay in the appendix or supplementary material, but
they should not be the headline result.

## Recommended Primary Outcomes

### 1. Action Clarity

1. **Rationale**
   Action Clarity captures whether the suggestion is understandable and specific
   enough to act on. In HCI, this matters because a suggestion that is novel but
   vague does not actually support decision-making.

2. **How to calculate**
   Use the existing 1-5 Likert ratings stored in
   `AI_Suggestions[*].ratings.Action_Clarity`.

   For the current JSON export:
   - aggregate by participant across their three sessions for each strategy,
   - compare `LLM-B`, `LLM-CF`, and `LLM-CT` with a within-participant omnibus
     test,
   - use paired post-hoc comparisons if the omnibus test is significant.

3. **Specific details**
   - Data source: downloaded JSON
   - Unit for inference: participant
   - Main comparison: prompting strategy
   - Timing comparison: `HAI` vs `AIH` when full data are available
   - Reporting: mean, SD, median, and inferential test

### 2. Feasibility

1. **Rationale**
   Feasibility captures whether the suggested action seems realistic for the
   user's actual situation. This is critical for co-planning systems because
   overly idealized suggestions do not help users improve future behavior.

2. **How to calculate**
   Use the existing 1-5 Likert ratings stored in
   `AI_Suggestions[*].ratings.Feasibility`.

   Analyze exactly as above:
   - participant-level aggregation,
   - within-participant comparison across prompt strategies,
   - timing comparison only when the full between-subject dataset is available.

3. **Specific details**
   - Keep Feasibility separate from Action Clarity in the main paper
   - If you later build a composite "actionability" score, first verify that
     Action Clarity and Feasibility hang together empirically
   - For UIST, I would still report both separately in the main figure/table

### 3. Goal Alignment

1. **Rationale**
   A suggestion can be clear and feasible but still miss the user's actual goal.
   Goal Alignment is therefore the most direct relevance metric for the system's
   planning quality.

2. **How to calculate**
   Use the existing 1-5 Likert ratings stored in
   `AI_Suggestions[*].ratings.Goal_Alignment`.

   Compare strategies within participant and, when the full dataset is
   available, estimate timing effects between `HAI` and `AIH`.

3. **Specific details**
   - Use participant as the inferential unit
   - Report this as a primary metric, not a secondary one
   - In discussion, interpret it as "fit to the user's stated objective," not
     as generic correctness

### 4. Insight / Novelty

1. **Rationale**
   This is the most paper-aligned user-rated metric because the abstract is
   fundamentally about augmenting counterfactual thinking. If AI is helping, it
   should expand the user's thinking rather than merely restate what they
   already knew.

2. **How to calculate**
   Use the existing 1-5 Likert ratings stored in
   `AI_Suggestions[*].ratings.Insight_Novelty`.

   Analyze within participant across strategies. When the full randomized sample
   is available, also test whether timing changes perceived novelty.

3. **Specific details**
   - This should be framed as a headline metric
   - Pair it with qualitative evidence from free-text feedback about "new ideas"
     or "different perspectives"
   - If one metric has to lead the user study story, I would choose this one

### 5. Human Counterfactual Expansion

1. **Rationale**
   This is the strongest metric for the paper's core claim. Ratings tell you how
   people felt about the suggestions, but they do not prove that human
   counterfactual thinking expanded. A direct before/after human-output metric
   does.

2. **How to calculate**
   Compare `User_Plans_Initial` and `User_Plans_Final` for each session.

   Recommended operationalizations:
   - count the number of distinct human plans before and after AI,
   - count the number of unique intervention points or actionable ideas before
     and after AI,
   - code plans into "what to change" and "how to change" categories, then
     compute the increase in unique categories after AI exposure.

3. **Specific details**
   - Data source: `User_Plans_Initial`, `User_Plans_Final`
   - Main outcome: delta from initial to final
   - Strongest paper framing: "breadth of human counterfactual space expanded"
   - This is a high-priority next analysis for the submission

### 6. AI Incorporation / Adoption into Final Plans

1. **Rationale**
   A co-planning system should not only present suggestions; it should also
   influence the human's final plan when appropriate. This metric measures
   whether the AI changed the user's output.

2. **How to calculate**
   For each session, compare AI suggestions with the final human plans.

   Recommended options:
   - manual coding: did the final human plan incorporate an AI-originated idea?
   - semantic matching: compare final plans with AI suggestions and estimate
     overlap,
   - selected-plan analysis: if `Final_Selected_PlanNumber` is reliable, use it
     to identify which plan the user ultimately chose.

3. **Specific details**
   - Data source: `AI_Suggestions`, `User_Plans_Final`, `Final_Selected_PlanNumber`
   - Best reported as an adoption rate or overlap score
   - This metric supports the claim that AI suggestions changed later reflection

### 7. Qualitative Design Theme Prevalence

1. **Rationale**
   UIST reviewers care about design implications. The free-text feedback can
   tell you what users wanted from the system beyond numeric ratings:
   personalization, interactivity, explanation, UI improvements, and so on.

2. **How to calculate**
   Use the top-level `feedback` fields.

   Recommended workflow:
   - extract responses,
   - manually code the text with at least two coders,
   - report prevalence of the final themes,
   - include representative quotes.

   The repository script can provide an LLM-assisted first-pass coding pass, but
   the final paper should still rely on human validation.

3. **Specific details**
   - Data source: top-level `feedback`
   - Current conservative interpretation:
     - `feedback_2` mostly reflects AI influence on thinking and decisions
     - `feedback_3` mostly reflects desired future system features
     - `feedback_4` is usually a URL and should likely be excluded
   - Report an intercoder agreement statistic if manually coded

### 8. Failure-Mode Prevalence

1. **Rationale**
   The most useful HCI discussion section often comes from concrete failures:
   repetition, low novelty, input burden, weak personalization, or limited
   interactivity.

2. **How to calculate**
   Code negative feedback mentions into a small set of failure themes, for
   example:
   - repetitive or derivative suggestions,
   - suggestions that restate the user's own ideas,
   - too much required input,
   - limited refinement or back-and-forth,
   - lack of context sensitivity.

3. **Specific details**
   - Use free-text responses plus, where helpful, low `Insight_Novelty` ratings
   - This is a design-implication metric, not just a complaint list
   - It is especially valuable if there are no large numeric differences between
     prompt strategies

## Recommended Analysis Structure

### Current JSON subset

- Use participant as the inferential unit
- Compare prompt strategies within participant
- Treat timing differences as descriptive only if the subset is unbalanced
- Use free-text feedback for thematic design implications

### Full paper dataset

If you recover the full `N = 128` trial dataset, the most defensible model is a
mixed-effects analysis:

- fixed effects: timing, prompt strategy, and their interaction
- random effects: participant, and scenario/session if identifiable

For ordinal ratings, an ordinal mixed model is ideal. If that is too heavy for
the paper timeline, participant-level aggregation plus nonparametric tests is an
acceptable fallback, but mixed models are stronger.

## What I Would Lead With in the Paper

If I were writing the UIST submission, I would emphasize metrics in this order:

1. Insight / Novelty
2. Human Counterfactual Expansion
3. AI Incorporation / Adoption
4. Action Clarity
5. Feasibility
6. Goal Alignment
7. Qualitative Design Themes
8. Failure Modes

## What I Would De-Emphasize

- Perplexity / fluency
- Generic automatic similarity scores
- Any metric that says the text "looks good" but does not show that human
  reflection improved

Those can still appear in the supplement, but they should not drive the main
story of this paper.
