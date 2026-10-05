library(dplyr)
library(readr)
library(stringdist)

setwd("~/Research/letterstringExperiment/study_philipp")

import_csvs <- function(folder_path) {
  # Get all CSV file paths in the folder
  csv_files <- list.files(
    path       = folder_path,
    pattern    = "\\.csv$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  if (length(csv_files) == 0) {
    stop("No CSV files found in: ", folder_path)
  }
  message("Found ", length(csv_files), " CSV file(s). Importing...")
  # Read and combine all CSVs
  combined <- csv_files |>
    lapply(read_csv, show_col_types = FALSE) |>
    bind_rows()
  message("Done. Combined dataset: ", nrow(combined), " rows x ", ncol(combined), " columns.")
  return(combined)
}

### import datasets add/rm necessary cols
df_items <- read.csv("mlc_letstr_items.csv") %>%
  distinct(across(-itemid), .keep_all = TRUE)
df_mlc <- import_csvs("results_mlc")  %>%
  rename(rowid = 1) %>%
  left_join(df_items) %>%
  mutate(provider = "Philipp Hellwig")
df_llm <- import_csvs("results_llms") %>%
  left_join(df_items) %>%
  mutate(pred = response) #%>%
df_llm <- df_llm %>%
  select(-all_of(setdiff(colnames(df_llm), colnames(df_mlc))))

### score mlc responses on levenstein dist from A, B, C and D
df_mlc <- df_mlc %>%
  mutate(lev_A = stringdist(A, pred, method = "lv"),
         lev_B = stringdist(B, pred, method = "lv"),
         lev_C = stringdist(C, pred, method = "lv"),
         lev_D = stringdist(D, pred, method = "lv"))
df <- bind_rows(df_llm, df_mlc)
df <- df |> filter(!is.na(transformation))

## visualize differences in acc by transformation
library(ggplot2)

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
  "claude-sonnet-4-20250514"                  = "Claude Sonnet 4",
  "gemini-2.5-flash"                          = "Gemini 2.5 Flash",
  "gpt-4.1"                                   = "GPT-4.1",
  "gpt-5.2"                                   = "GPT-5.2",
  "openai/gpt-oss-20b"                        = "GPT-OSS 20B",
  "openai/gpt-oss-120b"                       = "GPT-OSS 120B",
  "meta-llama/Llama-3.3-70B-Instruct-Turbo"   = "Llama 3.3 70B",
  "meta-llama/Llama-4-Scout-17B-16E-Instruct" = "Llama 4 Scout 17B",
  "mistralai/Ministral-3-14B-Instruct-2512"   = "Ministral 3 14B",
  "mistralai/Mistral-Small-24B-Instruct-2501" = "Mistral Small 24B",
  "nocopy_20"                                 = "NoCopy 20 alphabets",
  "copy_20"                                   = "Copy 20 alphabets",
  "copy_200"                                  = "Copy 200 alphabets"
)

df <- df |>
  mutate(
    transformation = factor(transformation, levels = names(transformation_map)),
    model          = factor(model, levels = rev(names(model_name_map)))
  )
df_plot <- df |>
  mutate(
    gen_type = dplyr::case_when(
      !grepl("copy", model, ignore.case = TRUE) ~ "other",  # LLMs / non-copy
      transformation %in% c("rr_sort", "extend_pred", "fix_interleave", "extend_group") ~ "compositional",
      transformation %in% c("reverse", "shift", "replicate") ~ "novel",
      TRUE ~ "seen"
    )
  )

ggplot(df_plot, aes(x = factor(model), y = correct, fill = gen_type)) +
  stat_summary(fun = mean, geom = "bar", color = "black") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  facet_wrap(~ transformation, labeller = labeller(transformation = transformation_map), ncol = 4, nrow = 5) +
  coord_flip() +  # horizontal bars
  #scale_fill_manual(values = c("TRUE" = "darkgreen", "FALSE" = "gray80"), guide = "none") +
  scale_fill_manual(
    values = c(
      seen          = "#2b6cb0",  # blue
      compositional = "#773399",  # purple
      novel         = "#d95f02",  # reddish orange
      other         = "gray80"    # LLMs / non-copy
    ),
    breaks = c("seen", "compositional", "novel"),
    labels = c("seen", "compositional", "novel"),
    name   = "Transformations"
  ) +
  scale_x_discrete(labels = model_name_map) +
  labs(
    title = "Accuracy by Model and Transformation",
    x     = "Model",
    y     = "Accuracy",
    fill  = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
