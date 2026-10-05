library(dplyr)
library(ggplot2)

# -------------------------------------------------
# 1.  Prepare ICL data (factorise needed columns)
# -------------------------------------------------
transformation_map <- c(
  extend_sequence       = "Extend (Ext)",
  succ                  = "Successor (Succ)",
  pred                  = "Predecessor (Pred)",
  remove_redundant      = "Remove Redundant (RmRed)",
  fix_alphabetic_seq    = "Fix",
  sort                  = "Sort",
  sort_group            = "Sort + Group",
  rr_interleave         = "RmRed + Interleave (IL)",
  rr_succ               = "RmRed + Succ",
  fix_extend            = "Fix + Ext",
  rr_sort               = "RmRed + Sort",
  extend_pred           = "Ext + Pred",
  fix_interleave        = "Fix + IL",
  extend_group          = "Ext + Group",
  extend_extend_succ    = "Ext + Ext + Succ",
  fix_pred_succ         = "Fix + Pred + Succ",
  reverse               = "Reverse",
  shift                 = "Shift",
  replicate             = "Replicate"
)

model_name_map <- c(
  "claude-sonnet-4-5-20250929"                = "Claude Sonnet 4.5",
  "claude-sonnet-4-6"                         = "Claude Sonnet 4.6",
  "gemini-2.5-flash"                          = "Gemini 2.5 Flash",
  "gemini-3-flash-preview"                    = "Gemini 3.0 Flash",
  "gpt-5.4"                                   = "GPT-5.4",
  "openai/gpt-oss-20b"                        = "GPT-OSS 20B",
  "openai/gpt-oss-120b"                       = "GPT-OSS 120B",
  "meta-llama/Llama-3.3-70B-Instruct-Turbo"   = "Llama 3.3 70B",
  "nocopy_20"                                 = "NoCopy 20 alphabets",
  "copy_20"                                   = "Copy 20 alphabets",
  "copy_200"                                  = "Copy 200 alphabets"
)

df_icl_llm <- df_icl_llm %>%
  mutate(
    transformation = factor(transformation,
                            levels = names(transformation_map)),
    model = factor(model,
                   levels = rev(names(model_name_map))),
    n_examples = factor(n_examples, levels = 0:5)   # make discrete
  )

# -------------------------------------------------
# 2.  Plot mean accuracy per model for each n_examples
# -------------------------------------------------
ggplot(df_icl_llm,
       aes(x = model,
           y = correct,
           fill = n_examples)) +                     # discrete fill
  stat_summary(fun = mean, geom = "bar", colour = "black") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  facet_wrap(~ n_examples,
             labeller = labeller(n_examples = function(x)
               paste0("n_examples = ", x)),
             ncol = 3) +                           # 2 rows × 3 columns
  coord_flip() +                                   # horizontal bars
  scale_fill_manual(
    values = c("#d73027", "#fc8d59", "#fee090",
               "#e0f3f8", "#91bfdb", "#4575b4"),
    name = "Number of examples"
  ) +
  scale_x_discrete(labels = model_name_map) +      # readable model names
  labs(
    title = "Accuracy by Model and Number of In‑Context Examples",
    x = "Model",
    y = "Mean Accuracy"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.y = element_text(size = 8)
  )

# -------------------------------------------------
# 3.  Summarise mean accuracy per model & n_examples
# -------------------------------------------------
df_summary <- df_icl_llm %>%
  mutate(
    n_examples = as.integer(as.character(n_examples))
  ) %>%
  group_by(model, n_examples) %>%
  summarise(
    mean_acc = mean(correct, na.rm = TRUE),
    sd_acc   = sd(correct, na.rm = TRUE),
    n        = sum(!is.na(correct)),
    se_acc   = sd_acc / sqrt(n),
    .groups  = "drop"
  ) %>%
  mutate(
    model = factor(model, levels = rev(names(model_name_map)))
  )

# optionally drop models that weren't reported in the original paper
df_summary_2 <- df_summary %>%
  filter(
    !model %in% c(
      "gpt-5.4",
      "claude-sonnet-4-6",
      "gemini-3-flash-preview"
    )
  ) %>%
  droplevels()

# -------------------------------------------------
# 4. compute MLC accs
# -------------------------------------------------

df_mlc_summary <- df_mlc %>%
  group_by(model) %>%
  summarise(
    mean_acc = mean(correct, na.rm = TRUE),
    sd_acc   = sd(correct, na.rm = TRUE),
    n        = sum(!is.na(correct)),
    se_acc   = sd_acc / sqrt(n),
    .groups  = "drop"
  )
# optionally drop models that weren't reported in the original paper
df_mlc_summary <- df_mlc_summary %>%
  filter(
    model %in% c("copy_200")
  ) %>%
  droplevels()

# -------------------------------------------------
# 5.  Plot accuracy improvement across examples
# -------------------------------------------------
# Plot mean accuracy with +/- 1 standard error
ggplot(df_summary, aes(x = n_examples, y = mean_acc, colour = model)) +
  geom_line(linewidth = 1) +
  geom_errorbar(
    aes(
      ymin = pmax(0, mean_acc - se_acc),
      ymax = pmin(1, mean_acc + se_acc)
    ),
    width = 0.12,
    linewidth = 0.5
  ) +
  geom_point(size = 2) +
  geom_hline(
    data = df_mlc_summary,
    aes(yintercept = mean_acc),
    inherit.aes = FALSE,
    colour = "black",
    linetype = "dotted",
    linewidth = 1
  ) +
  geom_text(
    data = df_mlc_summary,
    aes(
      x = 5,
      y = pmin(0.98, mean_acc + 0.02),
      label = "MLC copy 200"
    ),
    inherit.aes = FALSE,
    colour = "black",
    hjust = 1,
    vjust = 0,
    size = 3
  ) +
  scale_x_continuous(breaks = 0:5, name = "# In-context examples") +
  scale_y_continuous(name = "Mean Accuracy", limits = c(0, 1)) +
  scale_colour_manual(
    values = RColorBrewer::brewer.pal(
      n = dplyr::n_distinct(df_summary$model), 
      name = "Set3"
    ),
    labels = model_name_map,
    name = "Model"
  ) +
  labs(
    title = "Accuracy vs. Number of In-Context Examples per Model"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  )
