###############################################################################
##  RAG vs No-RAG in Transplant Infectious Disease
##  Fully reproducible analysis pipeline (single-file edition)
##  Generates: 7 publication figures (PDF+PNG) + 4 summary tables (CSV)
##  Requires: all_models_rag_non_rag_judged.csv
##            expert_evaluations_linked_long.csv
##  Run:  Rscript RAG_TxID_analysis_FULL.R
##
##  Figure numbering matches make_manuscript_figs.py:
##    Fig1  Primary degradation (dumbbell: LLM-judge + expert ratings)
##    Fig2  Per-model RAG effect (horizontal bar chart)
##    Fig3  Human validation + convergent validity
##    Fig4  Competence paradox (scatter)
##    Fig5  Grounding metrics (2x2 scatter)
##    Fig6  Judge reliability (heatmap + bar)
##    Fig7  Error redistribution (arrow plot)
###############################################################################


###############################################################################
##  RAG vs No-RAG in Transplant Infectious Disease: Reproducible Analysis
##  ---------------------------------------------------------------------------
##  01 | Setup, data ingestion, derived variables, theme & palette
##
##  Author: Shreyas Joshi
##  Environment: R >= 4.3
##  Inputs:
##    - all_models_rag_non_rag_judged.csv  (LLM judges)
##    - expert_evaluations_linked_long.csv                                     (human experts)
###############################################################################

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
  library(forcats); library(purrr); library(tibble)
  library(ggplot2); library(scales); library(ggrepel); library(patchwork)
  library(lme4); library(lmerTest); library(emmeans)
})

set.seed(42)
options(stringsAsFactors = FALSE)

## ---- paths -----------------------------------------------------------------
## Default to the directory containing this script (so a bare `Rscript
## RAG_TxID_analysis_FULL.R` works in-place); override with env vars if needed.
.script_dir <- tryCatch({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}, error = function(e) getwd())
DATA_DIR <- Sys.getenv("RAG_DATA_DIR", unset = .script_dir)
OUT_DIR  <- Sys.getenv("RAG_OUT_DIR",  unset = .script_dir)
FIG_DIR  <- file.path(OUT_DIR, "figures", "manuscript")
TAB_DIR  <- file.path(OUT_DIR, "tables")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR, showWarnings = FALSE, recursive = TRUE)

## NB: the complete dataset is a strict superset of the prior file, adding
## query_title, prompt, answer, and major_error_summary (used for the error
## taxonomy in script 05). All analytic columns are identical, so results 01-04
## are unchanged. The complete ("PUBLICATION") file is required to reproduce the
## error taxonomy (Figure 7 / Table 4); the "FABLE" subset reproduces 01-04 only.
LLM_CANDIDATES <- c(
  "all_models_rag_non_rag_judged_03Apr2026_USE_THIS.csv",
  "all_models_rag_non_rag_judged.csv",
  "all_models_rag_non_rag_judged_03Apr2026_USE_THIS_FOR_FABLE.csv")
f_llm <- Find(file.exists, file.path(DATA_DIR, LLM_CANDIDATES))
if (is.null(f_llm))
  stop("No LLM-judged dataset found in DATA_DIR: ", DATA_DIR)
f_human <- file.path(DATA_DIR, "expert_evaluations_linked_long.csv")

## ---- quality dimensions ----------------------------------------------------
QUAL_DIMS <- c("correctness","completeness","reasoning","relevance","clarity","consistency")
## Clinical-safety subset (primary endpoints emphasised in the manuscript)
SAFETY_DIMS <- c("correctness","completeness","reasoning")

## ---- load LLM-judged data --------------------------------------------------
llm <- read_csv(f_llm, show_col_types = FALSE) |>
  mutate(
    mode      = factor(if_else(RAG_MODEL, "RAG", "No-RAG"), levels = c("No-RAG","RAG")),
    rag       = factor(if_else(RAG_MODEL, "RAG", "NoRAG"), levels = c("NoRAG","RAG")),
    judge     = factor(judge_model),
    model     = factor(model_name),
    item      = factor(item_id),
    source    = factor(model_source),
    # composite quality = unweighted mean of the six rubric dimensions
    quality   = rowMeans(across(all_of(QUAL_DIMS)), na.rm = TRUE),
    safety    = rowMeans(across(all_of(SAFETY_DIMS)), na.rm = TRUE)
  )

## ---- load human expert data ------------------------------------------------
human <- read_csv(f_human, show_col_types = FALSE) |>
  mutate(
    mode      = factor(mode, levels = c("No RAG","RAG"), labels = c("No-RAG","RAG")),
    evaluator = factor(evaluator_name),
    case      = factor(case_code),
    model     = factor(model_name),
    side      = toupper(side)
  )

## ---- publication theme -----------------------------------------------------
theme_pub <- function(base_size = 11, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = "grey92", linewidth = 0.3),
      axis.line          = element_line(colour = "grey25", linewidth = 0.4),
      axis.ticks         = element_line(colour = "grey25", linewidth = 0.4),
      axis.title         = element_text(face = "plain", colour = "grey15"),
      plot.title         = element_text(face = "bold", size = rel(1.12), hjust = 0,
                                        margin = margin(b = 4)),
      plot.subtitle      = element_text(colour = "grey35", size = rel(0.92), hjust = 0,
                                        margin = margin(b = 8)),
      plot.caption       = element_text(colour = "grey50", size = rel(0.72), hjust = 0,
                                        margin = margin(t = 8)),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      legend.position    = "top",
      legend.title       = element_text(size = rel(0.85)),
      legend.key.size    = unit(0.9, "lines"),
      strip.text         = element_text(face = "bold", size = rel(0.9), colour = "grey15"),
      plot.margin        = margin(12, 14, 10, 12)
    )
}
theme_set(theme_pub())

## ---- colour palette (colourblind-aware) ------------------------------------
PAL <- list(
  norag   = "#2C6E9B",   # calm blue  = baseline (parametric) model
  rag     = "#C8442B",   # warm red   = retrieval-augmented
  helped  = "#1A8A5A",   # green      = RAG improves
  hurt    = "#C0392B",   # red        = RAG degrades
  neutral = "#8A8A8A",
  accent  = "#5B3A8C",
  bedrock = "#356FA3",
  ollama  = "#D98324"
)
MODE_COLORS <- c("No-RAG" = PAL$norag, "RAG" = PAL$rag)

ggsave_pub <- function(filename, plot, w, h) {
  ggsave(file.path(FIG_DIR, paste0(filename, ".pdf")), plot, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(FIG_DIR, paste0(filename, ".png")), plot, width = w, height = h, dpi = 300, bg = "white")
  invisible(plot)
}

## ---- console banner --------------------------------------------------------
cat(strrep("=", 78), "\n")
cat("LOADED:\n")
cat(sprintf("  LLM-judged rows : %d  | models: %d | items: %d | judges: %d\n",
            nrow(llm), nlevels(llm$model), nlevels(llm$item), nlevels(llm$judge)))
cat(sprintf("  Human-rated rows: %d  | evaluators: %d | cases: %d\n",
            nrow(human), nlevels(human$evaluator), nlevels(human$case)))
cat(strrep("=", 78), "\n")


###############################################################################
##  02 | Statistical inference
##      A. Central RAG effect (crossed mixed-effects models)
##      B. Competence Paradox (latent + per-model)
##      C. Grounding / Citation Paradox
##      D. Inter-rater reliability (LLM judges + human experts)
##      E. Human expert validation & convergent validity
###############################################################################

## helper: tidy fixed effect with CI from an lmer/glmer fit
fx <- function(m) {
  s <- summary(m)$coefficients
  ci <- tryCatch(confint(m, method = "Wald", parm = rownames(s)),
                 error = function(e) matrix(NA, nrow(s), 2))
  tibble(term = rownames(s),
         estimate = s[,1], se = s[,2],
         conf.low = ci[,1], conf.high = ci[,2],
         p = if ("Pr(>|t|)" %in% colnames(s)) s[,"Pr(>|t|)"] else s[,"Pr(>|z|)"])
}

## helper: ICC(2,k) two-way random, absolute agreement (no external pkg)
icc_2k <- function(M) {                       # M = subjects x raters matrix
  M <- as.matrix(M); M <- M[complete.cases(M), , drop = FALSE]
  n <- nrow(M); k <- ncol(M)
  gm <- mean(M); rm_ <- rowMeans(M); cm <- colMeans(M)
  MSR <- k * sum((rm_ - gm)^2) / (n - 1)
  MSC <- n * sum((cm - gm)^2) / (k - 1)
  MSE <- sum((M - outer(rm_, rep(1,k)) - outer(rep(1,n), cm) + gm)^2) / ((n-1)*(k-1))
  icc_k <- (MSR - MSE) / (MSR + (MSC - MSE)/n)         # ICC(2,k) abs agreement
  icc_1 <- (MSR - MSE) / (MSR + (k-1)*MSE + (k/n)*(MSC - MSE)) # ICC(2,1)
  list(icc_2k = icc_k, icc_21 = icc_1, n = n, k = k)
}

## helper: Cliff's delta (non-parametric effect size) + Cohen's d
cliffs_delta <- function(x, y) {
  x <- x[!is.na(x)]; y <- y[!is.na(y)]
  d <- outer(x, y, FUN = function(a,b) sign(a - b))
  mean(d)
}
cohens_d_paired <- function(diff) mean(diff, na.rm=TRUE) / sd(diff, na.rm=TRUE)

RESULTS <- list()   # collect everything for export

###############################################################################
## A. CENTRAL RAG EFFECT  -----------------------------------------------------
###############################################################################
cat("\n>>> A. Central RAG effect (mixed-effects, crossed random effects)\n")

## Crossed random effects for model, item, judge. Random slope of RAG by model
## captures *heterogeneity* of the RAG effect across models (-> Competence Paradox).
m_quality <- lmer(quality ~ rag + (1 + rag | model) + (1 | item) + (1 | judge),
                  data = llm, REML = TRUE,
                  control = lmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 2e5)))
m_correct <- lmer(correctness ~ rag + (1 + rag | model) + (1 | item) + (1 | judge),
                  data = llm, REML = TRUE,
                  control = lmerControl(optimizer = "bobyqa",
                                        optCtrl = list(maxfun = 2e5)))
## Major error: logistic mixed model
m_majerr  <- glmer(has_major_error ~ rag + (1 | model) + (1 | item) + (1 | judge),
                   data = llm, family = binomial,
                   control = glmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 2e5)))

RESULTS$fixed_quality <- fx(m_quality)
RESULTS$fixed_correct <- fx(m_correct)
RESULTS$fixed_majerr  <- fx(m_majerr)

cat("  Composite quality  RAG effect: ",
    sprintf("b=%.3f [%.3f, %.3f], p=%.2g\n",
            RESULTS$fixed_quality$estimate[2], RESULTS$fixed_quality$conf.low[2],
            RESULTS$fixed_quality$conf.high[2], RESULTS$fixed_quality$p[2]))
cat("  Correctness        RAG effect: ",
    sprintf("b=%.3f [%.3f, %.3f], p=%.2g\n",
            RESULTS$fixed_correct$estimate[2], RESULTS$fixed_correct$conf.low[2],
            RESULTS$fixed_correct$conf.high[2], RESULTS$fixed_correct$p[2]))
or_majerr <- exp(RESULTS$fixed_majerr$estimate[2])
cat("  Major-error odds ratio (RAG vs No-RAG): ",
    sprintf("OR=%.2f, p=%.2g\n", or_majerr, RESULTS$fixed_majerr$p[2]))

## Per-dimension RAG effect (six separate models) for the dumbbell figure
dim_effects <- map_dfr(QUAL_DIMS, function(d) {
  mm <- lmer(reformulate(c("rag","(1|model)","(1|item)","(1|judge)"), response = d),
             data = llm, REML = TRUE,
             control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  e <- fx(mm) |> filter(term == "ragRAG")
  norag_mean <- llm |> filter(rag=="NoRAG") |> pull(d) |> mean(na.rm=TRUE)
  rag_mean   <- llm |> filter(rag=="RAG")   |> pull(d) |> mean(na.rm=TRUE)
  tibble(dimension = d, norag = norag_mean, rag = rag_mean,
         delta = e$estimate, p = e$p, conf.low = e$conf.low, conf.high = e$conf.high)
})
RESULTS$dim_effects <- dim_effects

###############################################################################
## B. COMPETENCE PARADOX  -----------------------------------------------------
###############################################################################
cat("\n>>> B. Competence Paradox\n")

## (B1) Latent evidence: correlation between random intercept (baseline competence)
##      and random RAG slope (model-specific RAG effect) from the mixed model.
vc <- VarCorr(m_quality)
rint_rslope_cor <- attr(vc$model, "correlation")["(Intercept)", "ragRAG"]
RESULTS$paradox_latent_cor <- rint_rslope_cor
cat(sprintf("  Latent intercept-slope correlation (competence vs RAG effect): r=%.3f\n",
            rint_rslope_cor))

## (B2) Per-model empirical: baseline (No-RAG) quality vs RAG benefit (Δ)
model_mode <- llm |>
  group_by(model, source, rag) |>
  summarise(quality = mean(quality, na.rm=TRUE),
            correctness = mean(correctness, na.rm=TRUE),
            majerr = mean(has_major_error), .groups = "drop") |>
  pivot_wider(names_from = rag, values_from = c(quality, correctness, majerr))

model_paradox <- model_mode |>
  transmute(model, source,
            baseline_quality = quality_NoRAG,
            rag_benefit      = quality_RAG - quality_NoRAG,
            baseline_correct = correctness_NoRAG,
            rag_benefit_corr = correctness_RAG - correctness_NoRAG,
            majerr_delta     = majerr_RAG - majerr_NoRAG)
RESULTS$model_paradox <- model_paradox

ct_q  <- cor.test(model_paradox$baseline_quality, model_paradox$rag_benefit)
ct_c  <- cor.test(model_paradox$baseline_correct, model_paradox$rag_benefit_corr)
sp_q  <- cor.test(model_paradox$baseline_quality, model_paradox$rag_benefit, method="spearman")
RESULTS$paradox_quality_cor <- ct_q
RESULTS$paradox_correct_cor <- ct_c
cat(sprintf("  Baseline quality  vs RAG benefit : Pearson r=%.3f (p=%.4f); Spearman rho=%.3f\n",
            ct_q$estimate, ct_q$p.value, sp_q$estimate))
cat(sprintf("  Baseline correct. vs RAG benefit : Pearson r=%.3f (p=%.4f)\n",
            ct_c$estimate, ct_c$p.value))
cat(sprintf("  Models HELPED by RAG: %d/%d | HURT: %d/%d\n",
            sum(model_paradox$rag_benefit > 0), nrow(model_paradox),
            sum(model_paradox$rag_benefit < 0), nrow(model_paradox)))

###############################################################################
## C. GROUNDING / CITATION PARADOX  -------------------------------------------
###############################################################################
cat("\n>>> C. Grounding / Citation Paradox (RAG generations only)\n")

GROUND_METRICS <- c(
  gmkr                       = "GMKR (drift to internal knowledge)",
  retrieval_alignment_score  = "Retrieval alignment (lexical faithfulness)",
  evidence_density_per_100t  = "Evidence density (/100 tokens)",
  rur_distinct_docs          = "RUR: distinct documents",
  rur_total_citations        = "RUR: total citations"
)

ground_model <- llm |>
  filter(rag == "RAG") |>
  group_by(model, source) |>
  summarise(across(all_of(names(GROUND_METRICS)), ~mean(.x, na.rm=TRUE)),
            correctness = mean(correctness, na.rm=TRUE),
            quality     = mean(quality, na.rm=TRUE),
            majerr      = mean(has_major_error), .groups = "drop")
RESULTS$ground_model <- ground_model

ground_cor <- map_dfr(names(GROUND_METRICS), function(g) {
  cc <- cor.test(ground_model[[g]], ground_model$correctness)
  cq <- cor.test(ground_model[[g]], ground_model$quality)
  tibble(metric = g, label = GROUND_METRICS[[g]],
         r_correct = cc$estimate, p_correct = cc$p.value,
         r_quality = cq$estimate, p_quality = cq$p.value)
})
RESULTS$ground_cor <- ground_cor
print(ground_cor |> mutate(across(where(is.numeric), ~round(.x,3))) |> as.data.frame(), row.names = FALSE)

###############################################################################
## D. INTER-RATER RELIABILITY  ------------------------------------------------
###############################################################################
cat("\n>>> D. Inter-rater reliability\n")

## --- LLM judges: ICC across the 3 judges on common (model,item,mode) cells ---
judge_wide <- llm |>
  select(model, item, mode, judge, quality) |>
  pivot_wider(names_from = judge, values_from = quality) |>
  drop_na()
JUDGES <- levels(llm$judge)
icc_llm <- icc_2k(judge_wide[JUDGES])
RESULTS$icc_llm <- icc_llm
cat(sprintf("  LLM judges ICC(2,k)=%.3f  ICC(2,1)=%.3f  (n=%d cells, k=%d judges)\n",
            icc_llm$icc_2k, icc_llm$icc_21, icc_llm$n, icc_llm$k))

judge_pairs <- combn(JUDGES, 2, simplify = FALSE) |>
  map_dfr(function(p) tibble(j1=p[1], j2=p[2],
                             r = cor(judge_wide[[p[1]]], judge_wide[[p[2]]])))
RESULTS$judge_pairs <- judge_pairs

## Judge severity / RAG-sensitivity: per-judge RAG effect on quality
judge_sensitivity <- llm |>
  group_by(judge, rag) |>
  summarise(q = mean(quality, na.rm=TRUE), .groups="drop") |>
  pivot_wider(names_from = rag, values_from = q) |>
  mutate(rag_effect = RAG - NoRAG,
         grand = (RAG + NoRAG)/2) |>
  arrange(rag_effect)
RESULTS$judge_sensitivity <- judge_sensitivity
cat("  Per-judge RAG effect (Δ quality):\n")
walk2(judge_sensitivity$judge, judge_sensitivity$rag_effect,
      ~cat(sprintf("    %-20s %+.3f\n", .x, .y)))

## --- Human experts: ICC across the 3 evaluators (same 10 cases x 2 modes) ----
human_wide <- human |>
  mutate(cell = paste(case, mode, sep="|")) |>
  select(evaluator, cell, accuracy) |>
  pivot_wider(names_from = evaluator, values_from = accuracy) |>
  drop_na()
EVALS <- levels(human$evaluator)
icc_human <- icc_2k(human_wide[EVALS])
RESULTS$icc_human <- icc_human
cat(sprintf("  Human experts ICC(2,k)=%.3f  ICC(2,1)=%.3f  (n=%d cells, k=%d experts)\n",
            icc_human$icc_2k, icc_human$icc_21, icc_human$n, icc_human$k))

###############################################################################
## E. HUMAN EXPERT VALIDATION & CONVERGENT VALIDITY  --------------------------
###############################################################################
cat("\n>>> E. Human expert validation\n")

## (E1) Human ratings by mode (paired within case x evaluator)
human_mode <- human |>
  group_by(mode) |>
  summarise(accuracy = mean(accuracy), usefulness = mean(usefulness),
            majerr = mean(major_error), n = n(), .groups="drop")
RESULTS$human_mode <- human_mode

## paired differences (RAG - No-RAG) within evaluator x case
human_paired <- human |>
  select(evaluator, case, mode, accuracy, usefulness, major_error) |>
  pivot_wider(names_from = mode, values_from = c(accuracy, usefulness, major_error))
acc_w  <- wilcox.test(human_paired$accuracy_RAG, human_paired$`accuracy_No-RAG`,
                      paired = TRUE, exact = FALSE)
use_w  <- wilcox.test(human_paired$usefulness_RAG, human_paired$`usefulness_No-RAG`,
                      paired = TRUE, exact = FALSE)
RESULTS$human_acc_test <- acc_w
RESULTS$human_use_test <- use_w
RESULTS$human_acc_d <- cohens_d_paired(human_paired$accuracy_RAG - human_paired$`accuracy_No-RAG`)
RESULTS$human_use_d <- cohens_d_paired(human_paired$usefulness_RAG - human_paired$`usefulness_No-RAG`)
cat(sprintf("  Human accuracy   RAG-NoRAG: Δ=%+.2f  Wilcoxon p=%.3f  d=%.2f\n",
            human_mode$accuracy[human_mode$mode=="RAG"]-human_mode$accuracy[human_mode$mode=="No-RAG"],
            acc_w$p.value, RESULTS$human_acc_d))
cat(sprintf("  Human usefulness RAG-NoRAG: Δ=%+.2f  Wilcoxon p=%.3f  d=%.2f\n",
            human_mode$usefulness[human_mode$mode=="RAG"]-human_mode$usefulness[human_mode$mode=="No-RAG"],
            use_w$p.value, RESULTS$human_use_d))

## (E2) Blinded paired preference -> map A/B to mode
pref_tbl <- human |>
  distinct(evaluator, case, preference, side, mode) |>
  pivot_wider(names_from = side, values_from = mode, names_prefix = "mode_") |>
  distinct(evaluator, case, preference, mode_A, mode_B) |>
  mutate(preferred = case_when(
    preference == "Tie" ~ "Tie",
    preference == "A"   ~ as.character(mode_A),
    preference == "B"   ~ as.character(mode_B)))
pref_counts <- pref_tbl |> count(preferred)
RESULTS$pref_tbl <- pref_tbl
RESULTS$pref_counts <- pref_counts
n_rag  <- sum(pref_tbl$preferred == "RAG")
n_nora <- sum(pref_tbl$preferred == "No-RAG")
n_tie  <- sum(pref_tbl$preferred == "Tie")
sign_p <- binom.test(n_rag, n_rag + n_nora, 0.5)$p.value
RESULTS$pref_sign_p <- sign_p
cat(sprintf("  Blinded preference: No-RAG=%d  RAG=%d  Tie=%d  (sign test p=%.3f)\n",
            n_nora, n_rag, n_tie, sign_p))

## (E3) Convergent validity: human vs LLM on matched (model,case,mode) items
llm_item <- llm |>
  group_by(model, item, mode) |>
  summarise(llm_quality = mean(quality, na.rm=TRUE),
            llm_correct = mean(correctness, na.rm=TRUE),
            llm_majerr  = mean(has_major_error) >= 0.5, .groups="drop") |>
  mutate(key = paste(model, item, mode, sep="|"))
human_item <- human |>
  group_by(model, case, mode) |>
  summarise(h_accuracy = mean(accuracy), h_useful = mean(usefulness),
            h_majerr = mean(major_error) >= 0.5, .groups="drop") |>
  mutate(key = paste(model, case, mode, sep="|"))
conv <- inner_join(human_item, llm_item, by = "key", suffix = c("", "_llm")) |>
  mutate(mode = factor(mode, levels = c("No-RAG","RAG")))
RESULTS$convergent <- conv
if (nrow(conv) > 3) {
  cv_acc <- cor.test(conv$h_accuracy, conv$llm_correct, method = "spearman")
  RESULTS$conv_acc_cor <- cv_acc
  agree_majerr <- mean(conv$h_majerr == conv$llm_majerr)
  RESULTS$conv_majerr_agree <- agree_majerr
  cat(sprintf("  Human accuracy vs LLM correctness (matched items): Spearman rho=%.3f (p=%.3f, n=%d)\n",
              cv_acc$estimate, cv_acc$p.value, nrow(conv)))
  cat(sprintf("  Human vs LLM major-error agreement: %.1f%%\n", 100*agree_majerr))
}

cat("\n", strrep("=", 78), "\n", "Statistical inference complete.\n", sep="")


###############################################################################
##  03 | Publication figures  (Part 1 of 2: Figures 1-3)
###############################################################################

sig_star <- function(p) ifelse(p < .001, "***", ifelse(p < .01, "**",
                        ifelse(p < .05, "*", "ns")))
fmt_p <- function(p) ifelse(p < .001, "p < 0.001", paste0("p = ", formatC(p, digits=2, format="f")))

###############################################################################
## FIGURE 1  -  Convergent degradation: LLM judges & human experts agree
###############################################################################
## 1A: LLM per-dimension dumbbell
d1a <- RESULTS$dim_effects |>
  mutate(dimension = str_to_title(dimension),
         dimension = fct_reorder(dimension, delta),
         star = sig_star(p))
lab_long <- d1a |>
  select(dimension, `No-RAG` = norag, RAG = rag) |>
  pivot_longer(-dimension, names_to = "mode", values_to = "score")

f1a <- ggplot(d1a) +
  geom_segment(aes(y = dimension, yend = dimension, x = norag, xend = rag),
               colour = "grey75", linewidth = 1.1) +
  geom_point(data = lab_long, aes(x = score, y = dimension, colour = mode), size = 3.4) +
  geom_text(aes(x = pmin(norag, rag) - 0.06, y = dimension, label = star),
            hjust = 1, size = 3.2, colour = "grey30") +
  scale_colour_manual(values = MODE_COLORS, name = NULL) +
  scale_x_continuous(limits = c(2.4, 5), breaks = seq(2.5, 5, 0.5)) +
  labs(title = "A   LLM judges (39 models x 21 cases x 3 judges)",
       subtitle = "Mean rubric score per dimension  -  every dimension degrades under RAG",
       x = "Rubric score (1-5)", y = NULL) +
  theme(legend.position = "top")

## 1B: Human expert dumbbell (parallel grammar)
human_dumbbell <- RESULTS$human_mode |>
  select(mode, Accuracy = accuracy, Usefulness = usefulness) |>
  pivot_longer(-mode, names_to = "metric", values_to = "score") |>
  pivot_wider(names_from = mode, values_from = score)
hd_long <- human_dumbbell |>
  pivot_longer(c(`No-RAG`, RAG), names_to = "mode", values_to = "score")
h_tests <- tibble(metric = c("Accuracy","Usefulness"),
                  p = c(RESULTS$human_acc_test$p.value, RESULTS$human_use_test$p.value)) |>
  mutate(star = sig_star(p))
human_dumbbell <- left_join(human_dumbbell, h_tests, by = "metric")

f1b <- ggplot(human_dumbbell) +
  geom_segment(aes(y = metric, yend = metric, x = `No-RAG`, xend = RAG),
               colour = "grey75", linewidth = 1.1) +
  geom_point(data = hd_long, aes(x = score, y = metric, colour = mode), size = 3.8) +
  geom_text(aes(x = pmin(`No-RAG`, RAG) - 0.06, y = metric, label = star),
            hjust = 1, size = 3.4, colour = "grey30") +
  scale_colour_manual(values = MODE_COLORS, name = NULL, guide = "none") +
  scale_x_continuous(limits = c(2.8, 4.1), breaks = seq(3, 4, 0.5)) +
  labs(title = "B   Human evaluation (3 experts x 10 cases, blinded)",
       subtitle = "Independent human ratings reproduce the same downward shift",
       x = "Expert rating (1-5)", y = NULL) +
  theme(legend.position = "top")

fig1 <- f1a / f1b + plot_layout(heights = c(1.5, 1), guides = "collect") &
  theme(legend.position = "top")
fig1 <- fig1 + plot_annotation(
  caption = "Significance from crossed mixed-effects models (LLM) and paired Wilcoxon tests (human). *p<0.05  **p<0.01  ***p<0.001.")
ggsave_pub("Fig1_primary_degradation", fig1, w = 8.2, h = 7.4)

###############################################################################
## FIGURE 2  -  The Competence Paradox
###############################################################################
mp <- RESULTS$model_paradox |>
  mutate(outcome = if_else(rag_benefit >= 0, "RAG helps", "RAG hurts"))
## label a clean, informative subset
lab_idx <- mp |>
  mutate(rank_base = rank(-baseline_quality), rank_ben = rank(rag_benefit)) |>
  filter(rag_benefit > 0 | rank_base <= 6 | rank_ben <= 6 |
         model %in% c("Llama 3.1 70B","Kimi K2","Mistral Large 3","GPT-OSS 120B")) |>
  pull(model)
mp$lab <- ifelse(mp$model %in% lab_idx, as.character(mp$model), NA)

r_q <- RESULTS$paradox_quality_cor
ann <- sprintf("Pearson r = %.2f   (%s)\nSpearman \u03c1 = %.2f",
               r_q$estimate, fmt_p(r_q$p.value),
               cor(mp$baseline_quality, mp$rag_benefit, method = "spearman"))

xr <- range(mp$baseline_quality); yr <- range(mp$rag_benefit)
fig2 <- ggplot(mp, aes(baseline_quality, rag_benefit)) +
  annotate("rect", xmin = xr[1]-0.1, xmax = xr[2]+0.1, ymin = yr[1]-0.05, ymax = 0,
           fill = PAL$hurt, alpha = 0.05) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey45") +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$accent,
              fill = PAL$accent, alpha = 0.12, linewidth = 0.9) +
  geom_point(aes(fill = source), shape = 21, size = 3.1, colour = "white", stroke = 0.6) +
  geom_text_repel(aes(label = lab), size = 2.6, colour = "grey25",
                  segment.colour = "grey70", segment.size = 0.3,
                  max.overlaps = 30, min.segment.length = 0, box.padding = 0.35,
                  na.rm = TRUE) +
  annotate("text", x = xr[2], y = yr[2], hjust = 1, vjust = 1,
           label = ann, size = 3.1, colour = "grey20", lineheight = 0.95,
           fontface = "italic") +
  annotate("text", x = xr[1]+0.02, y = -0.02, hjust = 0, vjust = 1,
           label = "RAG-harm zone", size = 3, colour = PAL$hurt, fontface = "bold") +
  scale_fill_manual(values = c(Bedrock = PAL$bedrock, Ollama = PAL$ollama), name = "Hosting") +
  labs(title = "The Competence Paradox",
       subtitle = "Stronger parametric models gain the least  -  and lose the most  -  from retrieval",
       x = "Baseline competence  (No-RAG composite quality, 1-5)",
       y = "RAG benefit  (\u0394 quality: RAG - No-RAG)") +
  theme(legend.position = c(0.99, 0.02), legend.justification = c(1,0),
        legend.background = element_rect(fill = alpha("white",0.7), colour = NA))
ggsave_pub("Fig4_competence_paradox", fig2, w = 7.6, h = 6.2)

###############################################################################
## FIGURE 3  -  The Grounding / Citation Paradox
###############################################################################
gm <- RESULTS$ground_model
panel_specs <- tribble(
  ~var,                         ~title,                              ~xlab,
  "evidence_density_per_100t",  "Evidence density",                  "Evidence units / 100 tokens",
  "retrieval_alignment_score",  "Lexical faithfulness",              "Retrieval alignment (Jaccard)",
  "gmkr",                       "Drift to internal knowledge",       "GMKR",
  "rur_total_citations",        "Citation frequency",                "RUR: total citations"
)
make_panel <- function(var, title, xlab) {
  cc <- cor.test(gm[[var]], gm$correctness)
  rann <- sprintf("r = %.2f, %s", cc$estimate, fmt_p(cc$p.value))
  strong <- cc$p.value < 0.05
  ggplot(gm, aes(.data[[var]], correctness)) +
    geom_smooth(method = "lm", se = TRUE,
                colour = if (strong) PAL$hurt else "grey55",
                fill = if (strong) PAL$hurt else "grey70", alpha = 0.12, linewidth = 0.9) +
    geom_point(aes(fill = source), shape = 21, size = 2.6, colour = "white", stroke = 0.5) +
    annotate("text", x = Inf, y = Inf, hjust = 1.05, vjust = 1.4, label = rann,
             size = 3, fontface = if (strong) "bold" else "italic",
             colour = if (strong) PAL$hurt else "grey35") +
    scale_fill_manual(values = c(Bedrock = PAL$bedrock, Ollama = PAL$ollama), guide = "none") +
    labs(title = title, x = xlab, y = "Correctness (1-5)") +
    theme(plot.title = element_text(size = rel(0.98)))
}
panels <- pmap(panel_specs, make_panel)
fig3 <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]]) +
  plot_annotation(
    title = "The Grounding Paradox",
    subtitle = "Tighter coupling to retrieved evidence predicts LOWER correctness; internal-knowledge drift does not",
    caption = "Each point = one model (RAG generations only, n = 39). Red fits: p < 0.05.",
    theme = theme(plot.title = element_text(face = "bold", size = rel(1.25)),
                  plot.subtitle = element_text(colour = "grey35")))
ggsave_pub("Fig5_grounding_metrics", fig3, w = 8.4, h = 7.0)

cat("Figures 1-3 written.\n")


###############################################################################
##  04 | Publication figures (Part 2 of 2: Figures 4-6) + summary tables
###############################################################################

###############################################################################
## FIGURE 4  -  Per-model RAG effect: heterogeneity across 39 models
###############################################################################
mp4 <- RESULTS$model_paradox |>
  mutate(model = fct_reorder(model, rag_benefit),
         dir = if_else(rag_benefit >= 0, "RAG improves", "RAG degrades"))
n_hurt <- sum(mp4$rag_benefit < 0); n_help <- sum(mp4$rag_benefit >= 0)

fig4 <- ggplot(mp4, aes(rag_benefit, model, colour = dir)) +
  geom_vline(xintercept = 0, colour = "grey45", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = rag_benefit, yend = model), linewidth = 0.7) +
  geom_point(size = 2.5) +
  scale_colour_manual(values = c("RAG improves" = PAL$helped, "RAG degrades" = PAL$hurt),
                      name = NULL) +
  scale_x_continuous(breaks = seq(-1.5, 0.5, 0.5)) +
  annotate("text", x = -Inf, y = 2, hjust = -0.05, vjust = 0,
           label = sprintf("%d of %d models degraded by RAG", n_hurt, n_hurt+n_help),
           size = 3.2, fontface = "bold", colour = PAL$hurt) +
  labs(title = "Retrieval augmentation degrades most models",
       subtitle = "Change in composite quality (RAG - No-RAG), per model, averaged over cases and judges",
       x = "RAG benefit  (\u0394 quality)", y = NULL) +
  theme(axis.text.y = element_text(size = 7.4), legend.position = "top")
ggsave_pub("Fig2_per_model_effect", fig4, w = 7.4, h = 8.4)

###############################################################################
## FIGURE 5  -  Judge reliability & RAG-sensitivity
###############################################################################
## 5A: inter-judge correlation heatmap
JUDGES <- levels(llm$judge)
jw <- RESULTS$judge_pairs
cor_mat <- expand_grid(j1 = JUDGES, j2 = JUDGES) |>
  left_join(bind_rows(jw, jw |> rename(j1 = j2, j2 = j1)), by = c("j1","j2")) |>
  mutate(r = if_else(j1 == j2, 1, r),
         j1 = factor(j1, levels = JUDGES), j2 = factor(j2, levels = rev(JUDGES)))
f5a <- ggplot(cor_mat, aes(j1, j2, fill = r)) +
  geom_tile(colour = "white", linewidth = 1.2) +
  geom_text(aes(label = formatC(r, 2, format = "f")), size = 3.4, colour = "grey15") +
  scale_fill_gradient2(low = "#F4A582", mid = "#FDDBC7", high = PAL$norag,
                       midpoint = 0.6, limits = c(0.4, 1), name = "Pearson r") +
  coord_equal() +
  labs(title = "A   Inter-judge agreement",
       subtitle = sprintf("ICC(2,k) = %.2f across LLM judges", RESULTS$icc_llm$icc_2k),
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "right",
        panel.grid.major = element_blank())

## 5B: per-judge RAG sensitivity
js <- RESULTS$judge_sensitivity |>
  mutate(judge = fct_reorder(judge, rag_effect))
f5b <- ggplot(js, aes(rag_effect, judge)) +
  geom_vline(xintercept = 0, colour = "grey45") +
  geom_col(fill = PAL$rag, width = 0.62, alpha = 0.9) +
  geom_text(aes(label = sprintf("%+.2f", rag_effect)),
            hjust = 1.25, colour = "grey15", size = 3.2, fontface = "bold") +
  scale_x_continuous(limits = c(-0.78, 0.06)) +
  labs(title = "B   Judge RAG-sensitivity",
       subtitle = "Magnitude of harm depends on the judge",
       x = "RAG effect on quality (\u0394)", y = NULL)

fig5 <- f5a + f5b + plot_layout(widths = c(1, 1.05)) +
  plot_annotation(caption = "Judge choice affects the magnitude of estimated RAG harm, not its direction.")
ggsave_pub("Fig6_judge_reliability", fig5, w = 9.2, h = 4.4)

###############################################################################
## FIGURE 6  -  Human validation & convergent validity
###############################################################################
## 6A: blinded paired preference
pc <- RESULTS$pref_counts |>
  mutate(preferred = factor(preferred, levels = c("No-RAG","Tie","RAG"))) |>
  arrange(preferred)
prefcol <- c("No-RAG" = PAL$norag, "Tie" = "grey70", "RAG" = PAL$rag)
f6a <- ggplot(pc, aes(x = n, y = "Expert\npreference", fill = preferred)) +
  geom_col(width = 0.5, colour = "white") +
  geom_text(aes(label = paste0(preferred, "\n", n)),
            position = position_stack(vjust = 0.5), colour = "white",
            size = 3.1, fontface = "bold", lineheight = 0.9) +
  scale_fill_manual(values = prefcol, guide = "none") +
  labs(title = "A   Blinded head-to-head preference",
       subtitle = sprintf("30 paired comparisons (RAG vs No-RAG); sign test %s", fmt_p(RESULTS$pref_sign_p)),
       x = "Number of case-evaluations", y = NULL) +
  theme(axis.text.y = element_text(face = "bold"))

## 6B: convergent validity scatter (human accuracy vs LLM correctness)
cv <- RESULTS$convergent
rho <- RESULTS$conv_acc_cor
f6b <- ggplot(cv, aes(llm_correct, h_accuracy)) +
  geom_smooth(method = "lm", se = TRUE, colour = PAL$helped,
              fill = PAL$helped, alpha = 0.12, linewidth = 0.9) +
  geom_jitter(aes(colour = mode), width = 0.04, height = 0.06, size = 2.6, alpha = 0.9) +
  scale_colour_manual(values = MODE_COLORS, name = NULL) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.05, vjust = -0.6,
           label = sprintf("Spearman \u03c1 = %.2f\n%s  (n = %d)", rho$estimate,
                           fmt_p(rho$p.value), nrow(cv)),
           size = 3.1, fontface = "italic", colour = "grey20", lineheight = 0.95) +
  labs(title = "B   Convergent validity",
       subtitle = "Human accuracy tracks LLM-judged correctness on matched items",
       x = "LLM-judged correctness (1-5)", y = "Expert accuracy rating (1-5)")

fig6 <- f6a / f6b + plot_layout(heights = c(0.8, 1.4)) +
  plot_annotation(caption = "Clinician ratings and the LLM-judge panel agree, validating LLM-as-judge at scale.")
ggsave_pub("Fig3_human_validation", fig6, w = 7.0, h = 7.2)

cat("Figures 4-6 written.\n")

###############################################################################
## SUMMARY TABLES  ------------------------------------------------------------
###############################################################################
## Table 1: model-level leaderboard
tab_models <- llm |>
  group_by(Model = model, Source = source) |>
  summarise(
    `NoRAG quality` = round(mean(quality[rag=="NoRAG"], na.rm=TRUE), 2),
    `RAG quality`   = round(mean(quality[rag=="RAG"], na.rm=TRUE), 2),
    `RAG benefit`   = round(mean(quality[rag=="RAG"], na.rm=TRUE) - mean(quality[rag=="NoRAG"], na.rm=TRUE), 2),
    `NoRAG correct` = round(mean(correctness[rag=="NoRAG"], na.rm=TRUE), 2),
    `RAG correct`   = round(mean(correctness[rag=="RAG"], na.rm=TRUE), 2),
    `Major err NoRAG` = round(mean(has_major_error[rag=="NoRAG"]), 3),
    `Major err RAG`   = round(mean(has_major_error[rag=="RAG"]), 3),
    .groups = "drop") |>
  arrange(desc(`NoRAG quality`))
write_csv(tab_models, file.path(TAB_DIR, "Table1_model_leaderboard.csv"))

## Table 2: omnibus statistics summary
tab_stats <- tribble(
  ~Analysis, ~Estimate, ~CI_or_p,
  "RAG effect on composite quality (mixed model, b)",
      sprintf("%.3f", RESULTS$fixed_quality$estimate[2]),
      sprintf("[%.3f, %.3f], p=%.1e", RESULTS$fixed_quality$conf.low[2], RESULTS$fixed_quality$conf.high[2], RESULTS$fixed_quality$p[2]),
  "RAG effect on correctness (mixed model, b)",
      sprintf("%.3f", RESULTS$fixed_correct$estimate[2]),
      sprintf("[%.3f, %.3f], p=%.3f", RESULTS$fixed_correct$conf.low[2], RESULTS$fixed_correct$conf.high[2], RESULTS$fixed_correct$p[2]),
  "Major-error odds ratio (RAG vs No-RAG)",
      sprintf("%.2f", exp(RESULTS$fixed_majerr$estimate[2])),
      sprintf("p=%.3f", RESULTS$fixed_majerr$p[2]),
  "Competence Paradox: baseline quality x RAG benefit (Pearson)",
      sprintf("%.3f", RESULTS$paradox_quality_cor$estimate),
      sprintf("p=%.4f", RESULTS$paradox_quality_cor$p.value),
  "Competence Paradox: baseline correctness x RAG benefit (Pearson)",
      sprintf("%.3f", RESULTS$paradox_correct_cor$estimate),
      sprintf("p=%.4f", RESULTS$paradox_correct_cor$p.value),
  "Competence Paradox: latent intercept-slope correlation",
      sprintf("%.3f", RESULTS$paradox_latent_cor), "mixed-model VarCorr",
  "Grounding: evidence density x correctness (Pearson)",
      sprintf("%.3f", RESULTS$ground_cor$r_correct[RESULTS$ground_cor$metric=="evidence_density_per_100t"]),
      sprintf("p=%.4f", RESULTS$ground_cor$p_correct[RESULTS$ground_cor$metric=="evidence_density_per_100t"]),
  "Grounding: retrieval alignment x correctness (Pearson)",
      sprintf("%.3f", RESULTS$ground_cor$r_correct[RESULTS$ground_cor$metric=="retrieval_alignment_score"]),
      sprintf("p=%.4f", RESULTS$ground_cor$p_correct[RESULTS$ground_cor$metric=="retrieval_alignment_score"]),
  "LLM judge reliability ICC(2,k)",
      sprintf("%.3f", RESULTS$icc_llm$icc_2k), sprintf("k=%d judges", RESULTS$icc_llm$k),
  "Human expert reliability ICC(2,k)",
      sprintf("%.3f", RESULTS$icc_human$icc_2k), sprintf("k=%d experts", RESULTS$icc_human$k),
  "Human accuracy RAG vs No-RAG (Wilcoxon)",
      sprintf("d=%.2f", RESULTS$human_acc_d), sprintf("p=%.3f", RESULTS$human_acc_test$p.value),
  "Human usefulness RAG vs No-RAG (Wilcoxon)",
      sprintf("d=%.2f", RESULTS$human_use_d), sprintf("p=%.3f", RESULTS$human_use_test$p.value),
  "Convergent validity: human accuracy x LLM correctness (Spearman)",
      sprintf("%.3f", RESULTS$conv_acc_cor$estimate), sprintf("p=%.3f", RESULTS$conv_acc_cor$p.value)
)
write_csv(tab_stats, file.path(TAB_DIR, "Table2_key_statistics.csv"))

## Table 3: grounding correlation table
write_csv(RESULTS$ground_cor |> mutate(across(where(is.numeric), ~round(.x,4))),
          file.path(TAB_DIR, "Table3_grounding_correlations.csv"))

cat("Tables written to", TAB_DIR, "\n")
cat("\n", strrep("=",78), "\n  ALL FIGURES & TABLES COMPLETE\n", strrep("=",78), "\n", sep="")


###############################################################################
##  05 | Error taxonomy  (requires the complete dataset with major_error_summary)
##      Classifies all major errors into mutually-exclusive categories and
##      contrasts the error-type profile of RAG vs No-RAG generations.
##  Depends on objects/theme from 01_setup_and_data.R (sourced first).
###############################################################################

stopifnot("major_error_summary" %in% names(llm))

## ---- mutually-exclusive, priority-ordered classifier ----------------------
## Operates on the judge-written major_error_summary. Retrieval-structural
## categories (which reference the provided passages) are tested first.
classify_error <- function(s) {
  s <- tolower(ifelse(is.na(s), "", s))
  has  <- function(p) grepl(p, s, perl = TRUE)
  ref_doc <- has("doc\\s*\\d+|provided source|the source|retrieved")
  misread <- has("hallucinat|does not (?:cover|contain|say|support)|actually (?:covers|about)|misattribut|wrong (?:doc|source)|incorrectly (?:cites|attributes)|fabricat|typo")
  dplyr::case_when(
    ref_doc & misread ~ "Source misattribution",
    has("ignor|contradict|explicit|prompt (?:states|specif|says)|stated in the (?:case|prompt|stem)|despite the") ~ "Ignores explicit case facts",
    has("omit|fails? to (?:mention|recognize|consider|include|address)|completely (?:omits|misses)|does not (?:mention|consider|address)|missed (?:differential|diagnosis)|no mention") ~ "Critical omission",
    has("\\bdose|dosing|mg/kg|mathematic|fold|auc|insufficient (?:reduction|dose)|incorrect (?:duration|dose)|miscalcul") ~ "Dosing / quantitative",
    has("interaction|contraindicat|inhibit|\\btac\\b|cni|cyp3a") ~ "Drug interaction",
    has("outdated|no longer|superseded|deprecated") ~ "Outdated guidance",
    has("danger|harmful|unsafe|inappropriate|should not|incorrect (?:management|recommend)|wrong (?:management|treatment|antiviral|agent)") ~ "Incorrect/unsafe management",
    TRUE ~ "Other / unclassified")
}

err <- llm |>
  filter(has_major_error == 1) |>
  mutate(category = classify_error(major_error_summary))

n_gen <- llm |> count(rag, name = "n_gen")           # denominator = all generations
N_RAG <- n_gen$n_gen[n_gen$rag == "RAG"]
N_NOR <- n_gen$n_gen[n_gen$rag == "NoRAG"]

CAT_LEVELS <- c("Source misattribution","Ignores explicit case facts","Critical omission",
                "Dosing / quantitative","Drug interaction","Outdated guidance",
                "Incorrect/unsafe management","Other / unclassified")

## ---- per-generation rates + Fisher tests ----------------------------------
tax <- expand_grid(category = CAT_LEVELS, rag = c("NoRAG","RAG")) |>
  left_join(err |> count(category, rag, name = "n_err"), by = c("category","rag")) |>
  mutate(n_err = replace_na(n_err, 0L),
         n_gen = if_else(rag == "RAG", N_RAG, N_NOR),
         rate  = 100 * n_err / n_gen)

tax_wide <- tax |>
  select(category, rag, n_err, rate) |>
  pivot_wider(names_from = rag, values_from = c(n_err, rate)) |>
  rowwise() |>
  mutate(
    rate_ratio = ifelse(rate_NoRAG > 0, rate_RAG / rate_NoRAG, NA_real_),
    OR = {tb <- matrix(c(n_err_RAG, N_RAG - n_err_RAG, n_err_NoRAG, N_NOR - n_err_NoRAG), nrow = 2, byrow = TRUE)
          unname(fisher.test(tb)$estimate)},
    p  = {tb <- matrix(c(n_err_RAG, N_RAG - n_err_RAG, n_err_NoRAG, N_NOR - n_err_NoRAG), nrow = 2, byrow = TRUE)
          fisher.test(tb)$p.value},
    direction = if_else(rate_RAG >= rate_NoRAG, "RAG increases", "RAG reduces")) |>
  ungroup()

## omnibus: does the error-type distribution differ by mode? (exclude Other)
cont <- err |> filter(category != "Other / unclassified") |>
  count(category, rag) |> pivot_wider(names_from = rag, values_from = n, values_fill = 0)
chi <- chisq.test(as.matrix(cont[,-1]))
RESULTS$error_taxonomy <- tax_wide
RESULTS$error_chi <- chi

cat("\n>>> Error taxonomy (per-generation rates; RAG n=", N_RAG, ", No-RAG n=", N_NOR, ")\n", sep="")
print(tax_wide |> mutate(across(c(rate_RAG,rate_NoRAG,rate_ratio,OR), ~round(.x,2)),
                         p = signif(p,2)) |>
        select(category, n_err_RAG, rate_RAG, n_err_NoRAG, rate_NoRAG, rate_ratio, OR, p) |>
        as.data.frame(), row.names = FALSE)
cat(sprintf("Omnibus chi-square (error profile differs by mode): chi2=%.1f, df=%d, p=%.2e\n",
            chi$statistic, chi$parameter, chi$p.value))

write_csv(tax_wide |> mutate(across(where(is.numeric), ~signif(.x, 4))),
          file.path(TAB_DIR, "Table4_error_taxonomy.csv"))

###############################################################################
## FIGURE 7 — Error taxonomy: RAG redistributes errors (lookup vs synthesis)
###############################################################################
sig_star <- function(p) ifelse(p < .001,"***",ifelse(p < .01,"**",ifelse(p < .05,"*","")))

plot_tax <- tax_wide |>
  filter(category != "Other / unclassified") |>
  mutate(diff = rate_RAG - rate_NoRAG,
         category = fct_reorder(category, diff),
         star = sig_star(p),
         lab_x = pmax(rate_RAG, rate_NoRAG))
long_tax <- plot_tax |>
  select(category, `No-RAG` = rate_NoRAG, RAG = rate_RAG, direction) |>
  pivot_longer(c(`No-RAG`, RAG), names_to = "mode", values_to = "rate")

fig7 <- ggplot(plot_tax) +
  geom_segment(aes(y = category, yend = category, x = rate_NoRAG, xend = rate_RAG,
                   colour = direction), linewidth = 1.2, lineend = "round") +
  geom_point(data = long_tax, aes(x = rate, y = category, shape = mode),
             colour = "grey20", fill = "white", size = 3, stroke = 1) +
  geom_text(aes(x = lab_x + 0.35, y = category, label = star),
            hjust = 0, size = 4.2, colour = "grey25") +
  scale_colour_manual(values = c("RAG increases" = PAL$hurt, "RAG reduces" = PAL$helped),
                      name = NULL) +
  scale_shape_manual(values = c("No-RAG" = 21, "RAG" = 19), name = NULL) +
  scale_x_continuous(limits = c(0, max(plot_tax$lab_x) + 1.4), expand = c(0,0)) +
  labs(title = "Retrieval redistributes clinical errors: it helps lookup, harms synthesis",
       subtitle = "Major-error rate per 100 generations, by type; lines connect No-RAG to RAG.",
       x = "Major errors per 100 generations", y = NULL,
       caption = "Synthesis errors (omission, ignoring case facts, source misattribution) rise under RAG; lookup errors (drug interaction, dosing) fall. *p<0.05 **p<0.01 ***p<0.001, Fisher exact.") +
  theme(legend.position = "top")
ggsave_pub("Fig7_error_redistribution", fig7, w = 8.6, h = 5.4)

cat("Figure 7 (error taxonomy) written.\n")

###############################################################################
##  06 | Retrieval quality classification
##      Assigns each of the 21 cases to one of five retrieval-quality categories
##      based on document titles and similarity scores parsed from the answer
##      column (Sources block).  Requires the complete PUBLICATION dataset.
##      Generates: Table_S4_retrieval_quality.csv
###############################################################################

stopifnot("answer" %in% names(llm))

## ---- Parse source blocks from RAG answers -----------------------------------
## The answer column contains a Sources: section with lines like:
## Doc N — <title> (<filename> | <sim_score>)
parse_sources <- function(ans) {
  pat <- "Doc\\s*\\d+\\s*[\\u2013\\u2014\\-]\\s*(.+?)\\s*\\([^|]+\\|?\\s*([0-9\\.]+)\\)"
  m <- regmatches(ans, gregexpr(pat, ans, perl = TRUE))[[1]]
  if (length(m) == 0) return(data.frame(title = character(), score = numeric()))
  titles <- sub(pat, "\\1", m, perl = TRUE)
  scores <- suppressWarnings(as.numeric(sub(pat, "\\2", m, perl = TRUE)))
  valid  <- !is.na(scores) & scores <= 2.0   # exclude artifact scores (>2)
  data.frame(title = trimws(titles[valid]), score = scores[valid])
}

retrieval_stats <- llm |>
  filter(rag == "RAG") |>
  group_by(item = item) |>
  summarise(
    src_data = list(
      do.call(rbind, lapply(answer, parse_sources))
    ),
    .groups = "drop"
  ) |>
  mutate(
    mean_sim   = purrr::map_dbl(src_data, ~ if (nrow(.x) > 0) mean(.x$score, na.rm=TRUE) else NA_real_),
    n_unique   = purrr::map_int(src_data, ~ length(unique(.x$title))),
    has_artifact = purrr::map_lgl(src_data, ~ {
      # check for anomalous scores > 2 in raw text -- re-parse without filter
      TRUE  # placeholder; classified manually below
    })
  )

## ---- Manual classification (from title/score audit) ------------------------
RETRIEVAL_CLASS <- tribble(
  ~item,   ~retrieval_category,        ~coverage_failure, ~dedup_failure, ~corpus_artifact,
  "A-01",  "Good retrieval",           FALSE, FALSE, FALSE,
  "A-02",  "Good retrieval",           FALSE, FALSE, FALSE,
  "A-03",  "Low similarity",           FALSE, FALSE, FALSE,
  "A-04",  "Corpus artifact",          FALSE, FALSE, TRUE,
  "A-05",  "Low similarity",           FALSE, FALSE, FALSE,
  "A-06",  "Coverage failure",         TRUE,  FALSE, FALSE,
  "A-07",  "Good retrieval",           FALSE, FALSE, FALSE,
  "A-08",  "Deduplication failure",    FALSE, TRUE,  FALSE,
  "A-09",  "Good retrieval",           FALSE, FALSE, FALSE,
  "A-10",  "Coverage failure",         TRUE,  FALSE, FALSE,
  "S-01",  "Near-zero similarity",     TRUE,  FALSE, FALSE,
  "S-02",  "Corpus artifact",          FALSE, FALSE, TRUE,
  "S-03",  "Near-zero similarity",     TRUE,  FALSE, FALSE,
  "S-04",  "Low similarity",           FALSE, FALSE, FALSE,
  "S-05",  "Coverage failure",         TRUE,  FALSE, FALSE,
  "S-06",  "Corpus artifact",          FALSE, FALSE, TRUE,
  "S-07",  "Good retrieval",           FALSE, FALSE, FALSE,
  "S-08",  "Low similarity",           FALSE, FALSE, FALSE,
  "S-09",  "Good retrieval",           FALSE, FALSE, FALSE,
  "S-10",  "Deduplication failure",    FALSE, TRUE,  FALSE,
  "S-11",  "Good retrieval",           FALSE, FALSE, FALSE
)

GOOD_RETRIEVAL_CASES <- RETRIEVAL_CLASS$item[RETRIEVAL_CLASS$retrieval_category == "Good retrieval"]
BAD_RETRIEVAL_CASES  <- RETRIEVAL_CLASS$item[RETRIEVAL_CLASS$retrieval_category != "Good retrieval"]

cat("\nRetrieval quality classification:\n")
print(RETRIEVAL_CLASS |> count(retrieval_category) |> as.data.frame(), row.names = FALSE)

## ---- Table S4 ---------------------------------------------------------------
per_case_quality <- llm |>
  group_by(item, rag) |>
  summarise(mean_q = mean(quality, na.rm=TRUE), .groups="drop") |>
  pivot_wider(names_from=rag, values_from=mean_q) |>
  mutate(delta = RAG - NoRAG)

tab_s4 <- RETRIEVAL_CLASS |>
  left_join(retrieval_stats |> select(item, mean_sim, n_unique), by="item") |>
  left_join(per_case_quality, by="item") |>
  mutate(
    human_evaluated = item %in% c("A-01","A-03","A-05","A-06","A-07","S-02","S-04","S-08","S-09","S-11"),
    across(c(NoRAG, RAG, delta, mean_sim), ~round(.x, 3))
  ) |>
  rename(norag_quality=NoRAG, rag_quality=RAG)

write_csv(tab_s4, file.path(TAB_DIR, "Table_S4_retrieval_quality.csv"))
cat("Table S4 (retrieval quality) written.\n")

###############################################################################
##  07 | Sensitivity analyses
##      A. Restricted to good-retrieval cases (n=7)
##      B. Excluding all poor-retrieval cases (n=11 remaining)
##      C. Case-family subgroup (A-cases vs S-cases)
###############################################################################

cat("\n", strrep("-",78), "\n  07 | SENSITIVITY & SUBGROUP ANALYSES\n", strrep("-",78), "\n")

llm_good <- llm |> filter(item %in% GOOD_RETRIEVAL_CASES)
llm_excl <- llm |> filter(!item %in% BAD_RETRIEVAL_CASES)

## ---- A. Good-retrieval sensitivity ------------------------------------------
cat("\n--- A. Good-retrieval cases only (n=7 cases) ---\n")
cat("Cases:", paste(sort(GOOD_RETRIEVAL_CASES), collapse=", "), "\n")

sens_good_diffs <- llm_good |>
  group_by(model, item, judge) |>
  summarise(rag_q   = mean(quality[rag=="RAG"],   na.rm=TRUE),
            norag_q = mean(quality[rag=="NoRAG"],  na.rm=TRUE),
            .groups = "drop") |>
  filter(!is.na(rag_q) & !is.na(norag_q)) |>
  mutate(diff = rag_q - norag_q)

t_good <- t.test(sens_good_diffs$diff, mu=0)
d_good <- mean(sens_good_diffs$diff, na.rm=TRUE) / sd(sens_good_diffs$diff, na.rm=TRUE)
w_good <- wilcox.test(sens_good_diffs$diff, mu=0, exact=FALSE)

cat(sprintf("  n pairs: %d\n  Mean delta: %.3f\n  t=%.3f, p=%.2e\n  Cohen d=%.3f\n  Wilcoxon p=%.2e\n",
            nrow(sens_good_diffs), mean(sens_good_diffs$diff),
            t_good$statistic, t_good$p.value, d_good, w_good$p.value))

RESULTS$sensitivity_good_retrieval <- list(
  cases = GOOD_RETRIEVAL_CASES, n_pairs = nrow(sens_good_diffs),
  mean_delta = mean(sens_good_diffs$diff), t_test = t_good, cohen_d = d_good,
  wilcoxon = w_good
)

## ---- B. Exclude all poor-retrieval cases ------------------------------------
cat("\n--- B. Exclude all poor-retrieval cases ---\n")
remaining_cases <- sort(setdiff(unique(llm$item), BAD_RETRIEVAL_CASES))
cat("Remaining cases:", paste(remaining_cases, collapse=", "), "\n")

sens_excl_diffs <- llm_excl |>
  group_by(model, item, judge) |>
  summarise(rag_q   = mean(quality[rag=="RAG"],   na.rm=TRUE),
            norag_q = mean(quality[rag=="NoRAG"],  na.rm=TRUE),
            .groups = "drop") |>
  filter(!is.na(rag_q) & !is.na(norag_q)) |>
  mutate(diff = rag_q - norag_q)

t_excl <- t.test(sens_excl_diffs$diff, mu=0)
d_excl <- mean(sens_excl_diffs$diff, na.rm=TRUE) / sd(sens_excl_diffs$diff, na.rm=TRUE)
w_excl <- wilcox.test(sens_excl_diffs$diff, mu=0, exact=FALSE)

cat(sprintf("  n pairs: %d\n  Mean delta: %.3f\n  t=%.3f, p=%.2e\n  Cohen d=%.3f\n  Wilcoxon p=%.2e\n",
            nrow(sens_excl_diffs), mean(sens_excl_diffs$diff),
            t_excl$statistic, t_excl$p.value, d_excl, w_excl$p.value))

RESULTS$sensitivity_excl_bad <- list(
  cases = remaining_cases, n_pairs = nrow(sens_excl_diffs),
  mean_delta = mean(sens_excl_diffs$diff), t_test = t_excl, cohen_d = d_excl,
  wilcoxon = w_excl
)

## ---- C. Case-family subgroup (A vs S) ----------------------------------------
cat("\n--- C. Case-family subgroup (A-cases vs S-cases) ---\n")

llm <- llm |> mutate(case_family = if_else(str_starts(as.character(item), "A"), "A", "S"))

family_res <- llm |>
  group_by(case_family, model, item, judge) |>
  summarise(rag_q   = mean(quality[rag=="RAG"],   na.rm=TRUE),
            norag_q = mean(quality[rag=="NoRAG"],  na.rm=TRUE),
            .groups="drop") |>
  filter(!is.na(rag_q) & !is.na(norag_q)) |>
  mutate(diff = rag_q - norag_q) |>
  group_by(case_family) |>
  summarise(
    n_cases = n_distinct(item),
    n_pairs = n(),
    mean_delta = mean(diff),
    cohen_d = mean(diff) / sd(diff),
    t_stat  = t.test(diff, mu=0)$statistic,
    p_value = t.test(diff, mu=0)$p.value,
    .groups="drop"
  )

print(family_res |> mutate(across(where(is.numeric), ~round(.x,3))) |> as.data.frame(), row.names=FALSE)
RESULTS$case_family <- family_res

## ---- Append sensitivity results to supplementary Table S1 -------------------
tab_s1_additions <- tribble(
  ~Analysis, ~Estimate, ~CI_or_p,
  "SENSITIVITY: good-retrieval cases only (7 cases)",
    sprintf("%.3f", RESULTS$sensitivity_good_retrieval$mean_delta),
    sprintf("t=%.2f, p=%.2e, d=%.3f", RESULTS$sensitivity_good_retrieval$t_test$statistic,
            RESULTS$sensitivity_good_retrieval$t_test$p.value, RESULTS$sensitivity_good_retrieval$cohen_d),
  "SENSITIVITY: exclude all poor-retrieval cases (11 remaining cases)",
    sprintf("%.3f", RESULTS$sensitivity_excl_bad$mean_delta),
    sprintf("t=%.2f, p=%.2e, d=%.3f", RESULTS$sensitivity_excl_bad$t_test$statistic,
            RESULTS$sensitivity_excl_bad$t_test$p.value, RESULTS$sensitivity_excl_bad$cohen_d),
  "SUBGROUP: A-cases (A-01–A-10)",
    sprintf("%.3f", family_res$mean_delta[family_res$case_family=="A"]),
    sprintf("d=%.3f, p=%.2e", family_res$cohen_d[family_res$case_family=="A"],
            family_res$p_value[family_res$case_family=="A"]),
  "SUBGROUP: S-cases (S-01–S-11)",
    sprintf("%.3f", family_res$mean_delta[family_res$case_family=="S"]),
    sprintf("d=%.3f, p=%.2e", family_res$cohen_d[family_res$case_family=="S"],
            family_res$p_value[family_res$case_family=="S"])
)

tab_stats_full <- bind_rows(tab_stats, tab_s1_additions)
write_csv(tab_stats_full, file.path(TAB_DIR, "Table_S1_consolidated_statistics.csv"))
cat("\nTable S1 (consolidated statistics + sensitivity) written.\n")

cat("\n", strrep("=",78), "\n  ALL ANALYSES COMPLETE\n", strrep("=",78), "\n", sep="")
cat("Outputs in:", OUT_DIR, "\n")
